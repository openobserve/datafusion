// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Hash aggregation
use std::sync::Arc;

use crate::aggregates::group_values::{new_group_values, GroupValues};
use crate::aggregates::{
    evaluate_group_by, evaluate_many, AggregateMode, PhysicalGroupBy,
};
use crate::metrics::BaselineMetrics;
use crate::{aggregates, PhysicalExpr};

use arrow::array::*;
use arrow::datatypes::SchemaRef;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::memory_pool::proxy::VecAllocExt;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_execution::TaskContext;
use datafusion_expr::{EmitTo, GroupsAccumulator};
use datafusion_physical_expr::GroupsAccumulatorAdapter;

use super::order::GroupOrdering;
use super::AggregateExec;
use datafusion_physical_expr::aggregate::AggregateFunctionExpr;
use log::debug;

#[derive(Debug, Clone)]
/// This object tracks the aggregation phase (input/output)
pub(crate) enum ExecutionState {
    ReadingInput,
    /// When producing output, the remaining rows to output are stored
    /// here and are sliced off as needed in batch_size chunks
    ProducingOutput(RecordBatch),
    /// All input has been consumed and all groups have been emitted
    Done,
}

pub struct GroupedHashAggregateStream {
    // ========================================================================
    // PROPERTIES:
    // These fields are initialized at the start and remain constant throughout
    // the execution.
    // ========================================================================
    schema: SchemaRef,
    mode: AggregateMode,

    /// Arguments to pass to each accumulator.
    ///
    /// The arguments in `accumulator[i]` is passed `aggregate_arguments[i]`
    ///
    /// The argument to each accumulator is itself a `Vec` because
    /// some aggregates such as `CORR` can accept more than one
    /// argument.
    aggregate_arguments: Vec<Vec<Arc<dyn PhysicalExpr>>>,

    /// GROUP BY expressions
    group_by: PhysicalGroupBy,

    // ========================================================================
    // STATE FLAGS:
    // These fields will be updated during the execution. And control the flow of
    // the execution.
    // ========================================================================
    /// Tracks if this stream is generating input or output
    exec_state: ExecutionState,

    /// Have we seen the end of the input
    input_done: bool,

    // ========================================================================
    // STATE BUFFERS:
    // These fields will accumulate intermediate results during the execution.
    // ========================================================================
    /// An interning store of group keys
    group_values: Box<dyn GroupValues>,

    /// scratch space for the current input [`RecordBatch`] being
    /// processed. Reused across batches here to avoid reallocations
    current_group_indices: Vec<usize>,

    /// Accumulators, one for each `AggregateFunctionExpr` in the query
    ///
    /// For example, if the query has aggregates, `SUM(x)`,
    /// `COUNT(y)`, there will be two accumulators, each one
    /// specialized for that particular aggregate and its input types
    accumulators: Vec<Box<dyn GroupsAccumulator>>,

    // ========================================================================
    // EXECUTION RESOURCES:
    // Fields related to managing execution resources and monitoring performance.
    // ========================================================================
    /// The memory reservation for this grouping
    reservation: MemoryReservation,

    /// Execution metrics
    baseline_metrics: BaselineMetrics,
}

impl GroupedHashAggregateStream {
    /// Create a new GroupedHashAggregateStream
    pub fn new(agg: &AggregateExec, context: Arc<TaskContext>) -> Result<Self> {
        debug!("Creating GroupedHashAggregateStream");
        let agg_schema = agg.input().schema();
        let agg_group_by = agg.group_by.clone();

        let baseline_metrics = BaselineMetrics::new(&agg.metrics, 0);

        let timer = baseline_metrics.elapsed_compute().timer();

        let aggregate_exprs = agg.aggr_expr.clone();

        let aggregate_arguments = aggregates::aggregate_expressions(
            &agg.aggr_expr,
            &agg.mode,
            agg_group_by.num_group_exprs(),
        )?;

        // Instantiate the accumulators
        let accumulators: Vec<_> = aggregate_exprs
            .iter()
            .map(create_group_accumulator)
            .collect::<Result<_>>()?;

        let group_schema = agg_group_by.group_schema(&agg.input().schema())?;

        let name = format!("GroupedHashAggregateStream[merge_to_disk]");
        let reservation = MemoryConsumer::new(name)
            .with_can_spill(false)
            .register(context.memory_pool());

        let group_values = new_group_values(group_schema, &GroupOrdering::None)?;
        timer.done();

        let exec_state = ExecutionState::ReadingInput;

        Ok(GroupedHashAggregateStream {
            schema: agg_schema,
            mode: agg.mode,
            accumulators,
            aggregate_arguments,
            group_by: agg_group_by,
            reservation,
            group_values,
            current_group_indices: Default::default(),
            exec_state,
            baseline_metrics,
            input_done: false,
        })
    }

    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// Create an accumulator for `agg_expr` -- a [`GroupsAccumulator`] if
/// that is supported by the aggregate, or a
/// [`GroupsAccumulatorAdapter`] if not.
pub(crate) fn create_group_accumulator(
    agg_expr: &Arc<AggregateFunctionExpr>,
) -> Result<Box<dyn GroupsAccumulator>> {
    if agg_expr.groups_accumulator_supported() {
        agg_expr.create_groups_accumulator()
    } else {
        // Note in the log when the slow path is used
        debug!(
            "Creating GroupsAccumulatorAdapter for {}: {agg_expr:?}",
            agg_expr.name()
        );
        let agg_expr_captured = Arc::clone(agg_expr);
        let factory = move || agg_expr_captured.create_accumulator();
        Ok(Box::new(GroupsAccumulatorAdapter::new(factory)))
    }
}

impl GroupedHashAggregateStream {
    /// Perform group-by aggregation for the given [`RecordBatch`].
    pub fn group_aggregate_batch(&mut self, batch: RecordBatch) -> Result<()> {
        // Evaluate the grouping expressions
        let group_by_values = evaluate_group_by(&self.group_by, &batch)?;

        // Evaluate the aggregation expressions.
        let input_values = evaluate_many(&self.aggregate_arguments, &batch)?;

        for group_values in &group_by_values {
            // calculate the group indices for each input row
            self.group_values
                .intern(group_values, &mut self.current_group_indices)?;
            let group_indices = &self.current_group_indices;

            // Update ordering information if necessary
            let total_num_groups = self.group_values.len();

            // Gather the inputs to call the actual accumulator
            let t = self.accumulators.iter_mut().zip(input_values.iter());

            for (acc, values) in t {
                // Call the appropriate method on each aggregator with
                // the entire input row and the relevant group indexes
                match self.mode {
                    AggregateMode::Partial
                    | AggregateMode::Single
                    | AggregateMode::SinglePartitioned => {
                        acc.update_batch(values, group_indices, None, total_num_groups)?;
                    }
                    _ => {
                        // if aggregation is over intermediate states,
                        // use merge
                        acc.merge_batch(values, group_indices, None, total_num_groups)?;
                    }
                }
            }
        }

        match self.update_memory_reservation() {
            // Here we can ignore `insufficient_capacity_err` because we will spill later,
            // but at least one batch should fit in the memory
            Err(DataFusionError::ResourcesExhausted(_)) => Ok(()),
            other => other,
        }
    }

    fn update_memory_reservation(&mut self) -> Result<()> {
        let acc = self.accumulators.iter().map(|x| x.size()).sum::<usize>();
        let reservation_result = self.reservation.try_resize(
            acc + self.group_values.size() + self.current_group_indices.allocated_size(),
        );

        reservation_result
    }

    /// Create an output RecordBatch with the group keys and
    /// accumulator states/values specified in emit_to
    fn emit(&mut self, emit_to: EmitTo) -> Result<Option<RecordBatch>> {
        let schema = self.schema();
        if self.group_values.is_empty() {
            return Ok(None);
        }

        let mut output = self.group_values.emit(emit_to)?;

        // Next output each aggregate value
        for acc in self.accumulators.iter_mut() {
            output.extend(acc.state(emit_to)?);
        }

        // emit reduces the memory usage. Ignore Err from update_memory_reservation. Even if it is
        // over the target memory size after emission, we can emit again rather than returning Err.
        let _ = self.update_memory_reservation();
        let batch = RecordBatch::try_new(schema, output)?;
        debug_assert!(batch.num_rows() > 0);
        Ok(Some(batch))
    }

    /// Clear memory and shirk capacities to the size of the batch.
    fn clear_shrink(&mut self, batch: &RecordBatch) {
        self.group_values.clear_shrink(batch);
        self.current_group_indices.clear();
        self.current_group_indices.shrink_to(batch.num_rows());
    }

    /// Clear memory and shirk capacities to zero.
    fn clear_all(&mut self) {
        let s = self.schema();
        self.clear_shrink(&RecordBatch::new_empty(s));
    }

    /// common function for signalling end of processing of the input stream
    fn set_input_done_and_produce_output(&mut self) -> Result<()> {
        self.input_done = true;
        let elapsed_compute = self.baseline_metrics.elapsed_compute().clone();
        let timer = elapsed_compute.timer();
        let batch = self.emit(EmitTo::All)?;
        self.exec_state =
            batch.map_or(ExecutionState::Done, ExecutionState::ProducingOutput);
        timer.done();
        Ok(())
    }

    pub fn get_final_result(&mut self) -> Result<RecordBatch> {
        self.set_input_done_and_produce_output()?;
        let batch = match &self.exec_state {
            ExecutionState::ProducingOutput(batch) => batch.clone(),
            _ => panic!("Not producing output"),
        };
        self.clear_all();
        Ok(batch)
    }
}
