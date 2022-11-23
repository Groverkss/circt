//===- SimplexSchedulers.cpp - Linear programming-based schedulers --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of linear programming-based schedulers with a built-in simplex
// solver.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Utilities.h"

#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#include <algorithm>
#include <limits>

#define DEBUG_TYPE "simplex-schedulers"

using namespace circt;
using namespace circt::scheduling;
using namespace mlir::presburger;

namespace {

/// This class provides a framework to model certain scheduling problems as
/// lexico-parametric linear programs (LP), which are then solved with an
/// extended version of the dual simplex algorithm.
///
/// The approach is described in:
///  [1] B. D. de Dinechin, "Simplex Scheduling: More than Lifetime-Sensitive
///      Instruction Scheduling", PRISM 1994.22, 1994.
///  [2] B. D. de Dinechin, "Fast Modulo Scheduling Under the Simplex Scheduling
///      Framework", PRISM 1995.01, 1995.
///
/// Resource-free scheduling problems (called "central problems" in the papers)
/// have an *integer* linear programming formulation with a totally unimodular
/// constraint matrix. Such ILPs can however be solved optimally in polynomial
/// time with a (non-integer) LP solver (such as the simplex algorithm), as the
/// LP solution is guaranteed to be integer. Note that this is the same idea as
/// used by SDC-based schedulers.
///
/// Variables are expected to be added in order:
/// [II, objective variables, problem variables, constant]. The LexSimplex may
/// store them inside in some other order, but expects constraints to be added
/// in this variable order.
class SimplexSchedulerBase : protected LexSimplex {
protected:
  /// The objective is to minimize the start time of this operation.
  Operation *lastOp;

  /// Used to conveniently retrieve an operation's start time variable. The
  /// alternative would be to find the op's index in the problem's list of
  /// operations.
  DenseMap<Operation *, unsigned> startTimeVariables;

  // Number of objectives.
  const unsigned numObjectiveVariables;
  const unsigned numProblemVariables;

  /// Allow subclasses to collect additional constraints that are not part of
  /// the input problem, but should be modeled in the linear problem.
  SmallVector<Problem::Dependence> additionalConstraints;

  virtual Problem &getProblem() = 0;
  virtual void fillObjectiveRow(SmallVectorImpl<MPInt> &row, unsigned obj);
  virtual void fillConstraintRow(SmallVectorImpl<MPInt> &row,
                                 Problem::Dependence dep);
  virtual void fillAdditionalConstraintRow(SmallVectorImpl<MPInt> &row,
                                           Problem::Dependence dep);

  void buildTableau(unsigned initialII = 0);

  LogicalResult solveTableau();

  LogicalResult scheduleAt(unsigned startTimeVariable, unsigned timeStep);

  Fraction getVariable(unsigned startTimeVariable);
  Fraction getII() { return getVariable(0); }

  LogicalResult checkLastOp();

public:
  explicit SimplexSchedulerBase(Operation *lastOp, unsigned numObjectives,
                                unsigned numProblemVariables)
      : LexSimplex(1 + numObjectives + numProblemVariables), lastOp(lastOp),
        numObjectiveVariables(numObjectives),
        numProblemVariables(numProblemVariables) {}
  virtual LogicalResult schedule() = 0;
};

/// This class solves the basic, acyclic `Problem`.
class SimplexScheduler : public SimplexSchedulerBase {
private:
  Problem &prob;

protected:
  Problem &getProblem() override { return prob; }

public:
  SimplexScheduler(Problem &prob, Operation *lastOp)
      : SimplexSchedulerBase(lastOp, 1, prob.getOperations().size()),
        prob(prob) {}
  LogicalResult schedule() override;
};

/// This class solves the resource-free `CyclicProblem`.  The optimal initiation
/// interval (II) is determined as a side product of solving the parametric
/// problem, and corresponds to the "RecMII" (= recurrence-constrained minimum
/// II) usually considered as one component in the lower II bound used by modulo
/// schedulers.
class CyclicSimplexScheduler : public SimplexSchedulerBase {
private:
  CyclicProblem &prob;

protected:
  Problem &getProblem() override { return prob; }
  void fillConstraintRow(SmallVectorImpl<MPInt> &row,
                         Problem::Dependence dep) override;

  // ModuloSimplexScheduler uses the problem infrastructure but has a different
  // number of objectives.
  CyclicSimplexScheduler(CyclicProblem &prob, Operation *lastOp,
                         unsigned numObjectives)
      : SimplexSchedulerBase(lastOp, numObjectives,
                             prob.getOperations().size()),
        prob(prob) {}

public:
  CyclicSimplexScheduler(CyclicProblem &prob, Operation *lastOp)
      : SimplexSchedulerBase(lastOp, 1, prob.getOperations().size()),
        prob(prob) {}
  LogicalResult schedule() override;
};

// This class solves acyclic, resource-constrained `SharedOperatorsProblem` with
// a simplified version of the iterative heuristic presented in [2].
class SharedOperatorsSimplexScheduler : public SimplexSchedulerBase {
private:
  SharedOperatorsProblem &prob;

protected:
  Problem &getProblem() override { return prob; }

public:
  SharedOperatorsSimplexScheduler(SharedOperatorsProblem &prob,
                                  Operation *lastOp)
      : SimplexSchedulerBase(lastOp, 1, prob.getOperations().size()),
        prob(prob) {}
  LogicalResult schedule() override;
};

// This class solves the `ModuloProblem` using the iterative heuristic presented
// in [2].
class ModuloSimplexScheduler : public CyclicSimplexScheduler {
private:
  struct MRT {
    ModuloSimplexScheduler &sched;

    using TableType = SmallDenseMap<unsigned, DenseSet<Operation *>>;
    using ReverseTableType = SmallDenseMap<Operation *, unsigned>;
    SmallDenseMap<Problem::OperatorType, TableType> tables;
    SmallDenseMap<Problem::OperatorType, ReverseTableType> reverseTables;

    explicit MRT(ModuloSimplexScheduler &sched) : sched(sched) {}
    LogicalResult enter(Operation *op, unsigned timeStep, unsigned currentII);
    void release(Operation *op);
  };

  ModuloProblem &prob;
  SmallVector<unsigned> asapTimes, alapTimes;
  SmallVector<Operation *> unscheduled, scheduled;
  MRT mrt;

protected:
  Problem &getProblem() override { return prob; }
  enum { OBJ_LATENCY = 0, OBJ_AXAP = 1 /* i.e. either ASAP or ALAP */ };
  void fillObjectiveRow(SmallVectorImpl<MPInt> &row, unsigned obj) override;
  // void updateMargins();
  void incrementII(unsigned currentII);
  void scheduleOperation(Operation *n);

public:
  ModuloSimplexScheduler(ModuloProblem &prob, Operation *lastOp)
      : CyclicSimplexScheduler(prob, lastOp, /*numObjectives=*/2), prob(prob),
        mrt(*this) {}
  LogicalResult schedule() override;
};

// This class solves the `ChainingProblem` by relying on pre-computed
// chain-breaking constraints.
class ChainingSimplexScheduler : public SimplexSchedulerBase {
private:
  ChainingProblem &prob;
  float cycleTime;

protected:
  Problem &getProblem() override { return prob; }
  void fillAdditionalConstraintRow(SmallVectorImpl<MPInt> &row,
                                   Problem::Dependence dep) override;

public:
  ChainingSimplexScheduler(ChainingProblem &prob, Operation *lastOp,
                           float cycleTime)
      : SimplexSchedulerBase(lastOp, 1, prob.getOperations().size()),
        prob(prob), cycleTime(cycleTime) {}
  LogicalResult schedule() override;
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// SimplexSchedulerBase
//===----------------------------------------------------------------------===//

void SimplexSchedulerBase::fillObjectiveRow(SmallVectorImpl<MPInt> &row,
                                            unsigned obj) {
  // Minimize start time of user-specified last operation.
  // Constraint: objVar = startTimeVariables[lastOp];
  row[1 + obj] = 1; // The added 1 is due to the II offset.
  row[startTimeVariables[lastOp]] = -1;
}

void SimplexSchedulerBase::fillConstraintRow(SmallVectorImpl<MPInt> &row,
                                             Problem::Dependence dep) {
  // Constraint: dst >= src + latency.
  auto &prob = getProblem();
  Operation *src = dep.getSource();
  Operation *dst = dep.getDestination();
  unsigned latency = *prob.getLatency(*prob.getLinkedOperatorType(src));
  row.back() = -latency; // note the negation
  if (src != dst) { // note that these coefficients just zero out in self-arcs.
    row[startTimeVariables[src]] = -1;
    row[startTimeVariables[dst]] = 1;
  }
}

void SimplexSchedulerBase::fillAdditionalConstraintRow(
    SmallVectorImpl<MPInt> &row, Problem::Dependence dep) {
  // Handling is subclass-specific, so do nothing by default.
  (void)row;
  (void)dep;
}

void SimplexSchedulerBase::buildTableau(unsigned initialII) {
  auto &prob = getProblem();

  // Offsets for each variable type.
  unsigned problemVarOffset = numObjectiveVariables + 1;

  // Map each operation to a variable representing its start time.
  // The offset is II, objectives.
  unsigned var = problemVarOffset;
  for (Operation *op : prob.getOperations()) {
    startTimeVariables[op] = var;
    ++var;
  }

  // II, objectives, problem variables, constant.
  unsigned numCols = numObjectiveVariables + numProblemVariables + 2;
  SmallVector<MPInt, 8> row(numCols, MPInt(0));

  // II >= initialII.
  row[0] = 1;
  row.back() = -initialII;
  addInequality(row);
  row.back() = 0;
  row[0] = 0;

  // Make each objective variable positive.
  for (unsigned i = 0; i < numProblemVariables; ++i) {
    row[problemVarOffset + i] = 1;
    addInequality(row);
    row[problemVarOffset + i] = 0;
  }

  // Add objectives.
  for (unsigned i = 0; i < numObjectiveVariables; ++i) {
    fillObjectiveRow(row, i);
    addEquality(row);
    std::fill(row.begin(), row.end(), MPInt(0));
  }

  // Setup constraints for dependencies.
  for (auto *op : prob.getOperations()) {
    for (auto &dep : prob.getDependences(op)) {
      fillConstraintRow(row, dep);
      addInequality(row);
      std::fill(row.begin(), row.end(), MPInt(0));
    }
  }

  // Setup constraints for additional dependencies.
  for (auto &dep : additionalConstraints) {
    fillAdditionalConstraintRow(row, dep);
    addInequality(row);
    std::fill(row.begin(), row.end(), MPInt(0));
  }
}

LogicalResult SimplexSchedulerBase::solveTableau() {
  if (restoreRationalConsistency().failed()) {
    markEmpty();
    return failure();
  }

  Fraction ii = getII();
  if (ii.num % ii.den == 0) {
    // We have an integer solution.
    return success();
  }

  // We have a rational solution for II. Since II is satisfied for any II >=
  // ration solution, we can round up to the next integer and restore
  // consistency.
  SmallVector<MPInt, 8> row(getNumVariables() + 1, MPInt(0));
  row[0] = 1;
  row.back() = -ceil(ii);
  addInequality(row);

  LogicalResult res = restoreRationalConsistency();
  assert(res.succeeded() && "Rounded up II should be feasible");
  return success();
}

LogicalResult SimplexSchedulerBase::scheduleAt(unsigned startTimeVariable,
                                               unsigned timeStep) {

  // Take a snapshot of the simplex.
  unsigned snapshot = getSnapshot();

  // Set startTimeVariable to timeStep.
  SmallVector<MPInt, 8> row(getNumVariables() + 1, MPInt(0));
  row[startTimeVariable] = 1;
  row.back() = -timeStep;
  addEquality(row);

  auto solved = solveTableau();
  // If there is no solution, roll back.
  if (failed(solved)) {
    rollback(snapshot);
    return failure();
  }

  return success();
}

Fraction SimplexSchedulerBase::getVariable(unsigned startTimeVariable) {
  const Unknown &u = var[startTimeVariable];

  // If the variable is in column position, the sample value of M + x is
  // zero, so x = -M which is unbounded.
  assert(u.orientation == Orientation::Row && "The variable should be bounded");

  // If the variable is in row position, its sample value is the
  // entry in the constant column divided by the denominator.
  MPInt denom = tableau(u.pos, 0);
  if (usingBigM)
    if (tableau(u.pos, 2) != denom)
      llvm_unreachable("The variable should be bounded");
  return {tableau(u.pos, 1), denom};
}

LogicalResult SimplexSchedulerBase::checkLastOp() {
  auto &prob = getProblem();
  if (!prob.hasOperation(lastOp))
    return prob.getContainingOp()->emitError(
        "problem does not include last operation");
  return success();
}

//===----------------------------------------------------------------------===//
// SimplexScheduler
//===----------------------------------------------------------------------===//

LogicalResult SimplexScheduler::schedule() {
  if (failed(checkLastOp()))
    return failure();

  buildTableau(0);
  // Add contraint the II = 0.
  SmallVector<MPInt, 8> row(getNumVariables() + 1, MPInt(0));
  row[0] = 1;
  addEquality(row);

  // Find a feasible solution to the tableau.
  if (solveTableau().failed())
    return prob.getContainingOp()->emitError() << "problem is infeasible";

  for (auto *op : prob.getOperations())
    prob.setStartTime(
        op, int64_t(getVariable(startTimeVariables[op]).getAsInteger()));

  return success();
}

//===----------------------------------------------------------------------===//
// CyclicSimplexScheduler
//===----------------------------------------------------------------------===//

void CyclicSimplexScheduler::fillConstraintRow(SmallVectorImpl<MPInt> &row,
                                               Problem::Dependence dep) {
  SimplexSchedulerBase::fillConstraintRow(row, dep);
  if (auto dist = prob.getDistance(dep))
    row[0] = *dist;
}

LogicalResult CyclicSimplexScheduler::schedule() {
  if (failed(checkLastOp()))
    return failure();

  // Start with II of 1.
  buildTableau(1);

  // Find a feasible solution to the tableau.
  if (solveTableau().failed()) {
    return prob.getContainingOp()->emitError() << "problem is infeasible";
  }

  prob.setInitiationInterval(int64_t(getII().getAsInteger()));
  for (auto *op : prob.getOperations()) {
    prob.setStartTime(
        op, int64_t(getVariable(startTimeVariables[op]).getAsInteger()));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SharedOperatorsSimplexScheduler
//===----------------------------------------------------------------------===//

static bool isLimited(Operation *op, SharedOperatorsProblem &prob) {
  return prob.getLimit(*prob.getLinkedOperatorType(op)).value_or(0) > 0;
}

LogicalResult SharedOperatorsSimplexScheduler::schedule() {
  if (failed(checkLastOp()))
    return failure();

  buildTableau(0);
  // Add contraint the II = 0.
  SmallVector<MPInt, 8> row(getNumVariables() + 1, MPInt(0));
  row[0] = 1;
  addEquality(row);

  if (failed(solveTableau()))
    return prob.getContainingOp()->emitError() << "problem is infeasible";

  // The *heuristic* part of this scheduler starts here:
  // We will now *choose* start times for operations using a shared operator
  // type, in a way that respects the allocation limits, and consecutively solve
  // the LP with these added constraints. The individual LPs are still solved to
  // optimality (meaning: the start times of the "last" operation is still
  // optimal w.r.t. the already fixed operations), however the heuristic choice
  // means we cannot guarantee the optimality for the overall problem.

  // Determine which operations are subject to resource constraints.
  auto &ops = prob.getOperations();
  SmallVector<Operation *> limitedOps;
  for (auto *op : ops)
    if (isLimited(op, prob))
      limitedOps.push_back(op);

  // Build a priority list of the limited operations.
  //
  // We sort by the resource-free start times to produce a topological order of
  // the operations. Better priority functions are known, but require computing
  // additional properties, e.g. ASAP and ALAP times for mobility, or graph
  // analysis for height. Assigning operators (=resources) in this order at
  // least ensures that the (acyclic!) problem remains feasible throughout the
  // process.
  //
  // TODO: Implement more sophisticated priority function.
  std::stable_sort(limitedOps.begin(), limitedOps.end(),
                   [&](Operation *a, Operation *b) {
                     return getVariable(startTimeVariables[a]) <
                            getVariable(startTimeVariables[b]);
                   });

  // Store the number of operations using an operator type in a particular time
  // step.
  SmallDenseMap<Problem::OperatorType, SmallDenseMap<unsigned, unsigned>>
      reservationTable;

  for (auto *op : limitedOps) {
    auto opr = *prob.getLinkedOperatorType(op);
    unsigned limit = prob.getLimit(opr).value_or(0);
    assert(limit > 0);

    // Find the first time step (beginning at the current start time in the
    // partial schedule) in which an operator instance is available.
    unsigned startTimeVar = startTimeVariables[op];
    unsigned candTime = int64_t(getVariable(startTimeVar).getAsInteger());
    while (reservationTable[opr].lookup(candTime) == limit)
      ++candTime;

    // Fix the start time. As explained above, this cannot make the problem
    // infeasible.
    auto fixed = scheduleAt(startTimeVar, candTime);
    assert(succeeded(fixed));
    (void)fixed;

    // Record the operator use.
    ++reservationTable[opr][candTime];
  }

  for (auto *op : prob.getOperations()) {
    prob.setStartTime(
        op, int64_t(getVariable(startTimeVariables[op]).getAsInteger()));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ModuloSimplexScheduler
//===----------------------------------------------------------------------===//

LogicalResult ModuloSimplexScheduler::MRT::enter(Operation *op,
                                                 unsigned timeStep,
                                                 unsigned currentII) {
  auto opr = *sched.prob.getLinkedOperatorType(op);
  auto lim = *sched.prob.getLimit(opr);
  assert(lim > 0);

  auto &revTab = reverseTables[opr];
  assert(!revTab.count(op));

  unsigned slot = timeStep % currentII;
  auto &cell = tables[opr][slot];
  if (cell.size() < lim) {
    cell.insert(op);
    revTab[op] = slot;
    return success();
  }
  return failure();
}

void ModuloSimplexScheduler::MRT::release(Operation *op) {
  auto opr = *sched.prob.getLinkedOperatorType(op);
  auto &revTab = reverseTables[opr];
  auto it = revTab.find(op);
  assert(it != revTab.end());
  tables[opr][it->second].erase(op);
  revTab.erase(it);
}

void ModuloSimplexScheduler::fillObjectiveRow(SmallVectorImpl<MPInt> &row,
                                              unsigned obj) {
  switch (obj) {
  case OBJ_LATENCY:
    // Minimize start time of user-specified last operation.
    row[startTimeVariables[lastOp]] = 1;
    return;
  case OBJ_AXAP:
    // Minimize sum of start times of all-but-the-last operation.
    for (auto *op : getProblem().getOperations())
      if (op != lastOp)
        row[startTimeVariables[op]] = 1;
    return;
  }
}

// void ModuloSimplexScheduler::updateMargins() {
//   // Assumption: current secondary objective is "ASAP".
//   // Negate the objective row once to effectively maximize the sum of start
//   // times, which yields the "ALAP" times after solving the tableau. Then,
//   // negate it again to restore the "ASAP" objective, and store these times
//   as
//   // well.
//   for (auto *axapTimes : {&alapTimes, &asapTimes}) {
//     multiplyRow(OBJ_AXAP, -1);
//     // This should not fail for a feasible tableau.
//     auto dualFeasRestored = restoreDualFeasibility();
//     auto solved = solveTableau();
//     assert(succeeded(dualFeasRestored) && succeeded(solved));
//     (void)dualFeasRestored, (void)solved;
//
//     for (unsigned stv = 0; stv < startTimeLocations.size(); ++stv)
//       (*axapTimes)[stv] = getStartTime(stv);
//   }
// }

void ModuloSimplexScheduler::incrementII(unsigned currentII) {
  // Add constraint that II >= currentII + 1.
  SmallVector<MPInt> row(getNumVariables() + 1, MPInt(0));
  row[0] = 1;
  row.back() = -currentII - 1;
  addInequality(row);

  LogicalResult res = solveTableau();
  assert(res.succeeded() && "Next II has to be feasible.");
}

void ModuloSimplexScheduler::scheduleOperation(Operation *n) {
  unsigned stvN = startTimeVariables[n];

  // Get current state of the LP, and determine range of alternative times
  // guaranteed to be feasible.
  unsigned stN = getStartTime(stvN);
  unsigned lbN = (unsigned)std::max<int>(asapTimes[stvN], stN - parameterT + 1);
  unsigned ubN = (unsigned)std::min<int>(alapTimes[stvN], lbN + parameterT - 1);

  LLVM_DEBUG(dbgs() << "Attempting to schedule at t=" << stN << ", or in ["
                    << lbN << ", " << ubN << "]: " << *n << '\n');

  SmallVector<unsigned> candTimes;
  candTimes.push_back(stN);
  for (unsigned ct = lbN; ct <= ubN; ++ct)
    if (ct != stN)
      candTimes.push_back(ct);

  for (unsigned ct : candTimes) {
    if (succeeded(mrt.enter(n, ct))) {
      auto fixedN = scheduleAt(stvN, stN);
      assert(succeeded(fixedN));
      (void)fixedN;
      LLVM_DEBUG(dbgs() << "Success at t=" << stN << " " << *n << '\n');
      return;
    }
  }

  // As a last resort, increase II to make room for the op. De Dinechin's
  // Theorem 1 lays out conditions/guidelines to transform the current partial
  // schedule for II to a valid one for a larger II'.

  LLVM_DEBUG(dbgs() << "Incrementing II to " << (parameterT + 1)
                    << " to resolve resource conflict for " << *n << '\n');

  // Note that the approach below is much simpler than in the paper
  // because of the fully-pipelined operators. In our case, it's always
  // sufficient to increment the II by one.

  // Decompose start time.
  unsigned phiN = stN / parameterT;
  unsigned tauN = stN % parameterT;

  // Keep track whether the following moves free at least one operator
  // instance in the slot desired by the current op - then it can stay there.
  unsigned deltaN = 1;

  // We're going to revisit the current partial schedule.
  SmallVector<Operation *> moved;
  for (Operation *j : scheduled) {
    unsigned stvJ = startTimeVariables[j];
    unsigned stJ = getStartTime(stvJ);
    unsigned phiJ = stJ / parameterT;
    unsigned tauJ = stJ % parameterT;
    unsigned deltaJ = 0;

    // To actually resolve the resource conflicts, we move operations that are
    // "preceded" (cf. de Dinechin's ≺ relation) one slot to the right.
    if (tauN < tauJ || (tauN == tauJ && phiN > phiJ) ||
        (tauN == tauJ && phiN == phiJ && stvN < stvJ)) {
      // TODO: Replace the last condition with a proper graph analysis.

      deltaJ = 1;
      moved.push_back(j);
      if (tauN == tauJ)
        deltaN = 0;
    }

    // Apply the move to the tableau.
    moveBy(stvJ, deltaJ);
  }

  // Finally, increment the II.
  incrementII();
  auto solved = solveTableau();
  assert(succeeded(solved));
  (void)solved;

  // Re-enter moved operations into their new slots.
  for (auto *m : moved)
    mrt.release(m);
  for (auto *m : moved) {
    auto enteredM = mrt.enter(m, getStartTime(startTimeVariables[m]));
    assert(succeeded(enteredM));
    (void)enteredM;
  }

  // Finally, schedule the operation. Adding `phiN` accounts for the implicit
  // shift caused by incrementing the II; cf. `incrementII()`.
  auto fixedN = scheduleAt(stvN, stN + phiN + deltaN);
  auto enteredN = mrt.enter(n, tauN + deltaN);
  assert(succeeded(fixedN) && succeeded(enteredN));
  (void)fixedN, (void)enteredN;
}

LogicalResult ModuloSimplexScheduler::schedule() {
  return failure();
  // if (failed(checkLastOp()))
  //   return failure();

  // parameterS = 0;
  // parameterT = 1;
  // buildTableau();
  // asapTimes.resize(startTimeLocations.size());
  // alapTimes.resize(startTimeLocations.size());

  // LLVM_DEBUG(dbgs() << "Initial tableau:\n"; dumpTableau());

  // if (failed(solveTableau()))
  //   return prob.getContainingOp()->emitError() << "problem is infeasible";

  // // Determine which operations are subject to resource constraints.
  // auto &ops = prob.getOperations();
  // for (auto *op : ops)
  //   if (isLimited(op, prob))
  //     unscheduled.push_back(op);

  // // Main loop: Iteratively fix limited operations to time steps.
  // while (!unscheduled.empty()) {
  //   // Update ASAP/ALAP times.
  //   updateMargins();

  //   // Heuristically (here: least amount of slack) pick the next operation
  //   to
  //   // schedule.
  //   auto *opIt =
  //       std::min_element(unscheduled.begin(), unscheduled.end(),
  //                        [&](Operation *opA, Operation *opB) {
  //                          auto stvA = startTimeVariables[opA];
  //                          auto stvB = startTimeVariables[opB];
  //                          auto slackA = alapTimes[stvA] - asapTimes[stvA];
  //                          auto slackB = alapTimes[stvB] - asapTimes[stvB];
  //                          return slackA < slackB;
  //                        });
  //   Operation *op = *opIt;
  //   unscheduled.erase(opIt);

  //   scheduleOperation(op);
  //   scheduled.push_back(op);
  // }

  // LLVM_DEBUG(dbgs() << "Final tableau:\n"; dumpTableau();
  //            dbgs() << "Solution found with II = " << parameterT
  //                   << " and start time of last operation = "
  //                   << -getParametricConstant(0) << '\n');

  // prob.setInitiationInterval(parameterT);
  // for (auto *op : ops)
  //   prob.setStartTime(op, getStartTime(startTimeVariables[op]));

  // return success();
}

//===----------------------------------------------------------------------===//
// ChainingSimplexScheduler
//===----------------------------------------------------------------------===//

void ChainingSimplexScheduler::fillAdditionalConstraintRow(
    SmallVectorImpl<MPInt> &row, Problem::Dependence dep) {
  fillConstraintRow(row, dep);
  // One _extra_ time step breaks the chain.
  row.back() -= 1;
}

LogicalResult ChainingSimplexScheduler::schedule() {
  if (failed(checkLastOp()) || failed(computeChainBreakingDependences(
                                   prob, cycleTime, additionalConstraints)))
    return failure();

  buildTableau(0);
  // Add contraint the II = 0.
  SmallVector<MPInt, 8> row(getNumVariables() + 1, MPInt(0));
  row[0] = 1;
  addEquality(row);

  if (failed(solveTableau()))
    return prob.getContainingOp()->emitError() << "problem is infeasible ";

  for (auto *op : prob.getOperations())
    prob.setStartTime(
        op, int64_t(getVariable(startTimeVariables[op]).getAsInteger()));

  auto filledIn = computeStartTimesInCycle(prob);
  assert(succeeded(filledIn)); // Problem is known to be acyclic at this point.
  (void)filledIn;

  return success();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

LogicalResult scheduling::scheduleSimplex(Problem &prob, Operation *lastOp) {
  SimplexScheduler simplex(prob, lastOp);
  return simplex.schedule();
}

LogicalResult scheduling::scheduleSimplex(CyclicProblem &prob,
                                          Operation *lastOp) {
  CyclicSimplexScheduler simplex(prob, lastOp);
  return simplex.schedule();
}

LogicalResult scheduling::scheduleSimplex(SharedOperatorsProblem &prob,
                                          Operation *lastOp) {
  SharedOperatorsSimplexScheduler simplex(prob, lastOp);
  return simplex.schedule();
}

LogicalResult scheduling::scheduleSimplex(ModuloProblem &prob,
                                          Operation *lastOp) {
  ModuloSimplexScheduler simplex(prob, lastOp);
  return simplex.schedule();
}

LogicalResult scheduling::scheduleSimplex(ChainingProblem &prob,
                                          Operation *lastOp, float cycleTime) {
  ChainingSimplexScheduler simplex(prob, lastOp, cycleTime);
  return simplex.schedule();
}
