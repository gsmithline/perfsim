"""Corrected NDlib HK iteration. Upstream HKModel.iteration computes N node
updates but writes only the last one back (the write sits outside the loop),
so one call moves a single node. Subclassing is impossible (their __init__
uses super(self.__class__, ...), which recurses), so we patch the method:
same asynchronous update rule, every update written. One iteration() = one
true sweep of N random node updates.
"""

import numpy as np

import ndlib.models.opinions as op


def _fixed_iteration(self, node_status=True):
    self.clean_initial_status(None)
    if self.actual_iteration == 0:
        self.actual_iteration += 1
        return {"iteration": 0, "status": {}}
    actual = dict(self.status)
    eps = self.params["model"]["epsilon"]
    nodes = list(self.graph.nodes)
    for _ in range(len(nodes)):
        n1 = nodes[np.random.randint(len(nodes))]
        close = [actual[v] for v in self.graph.neighbors(n1)
                 if abs(actual[n1] - actual[v]) < eps]
        if close:
            actual[n1] = sum(close) / len(close)
    self.status = actual
    self.actual_iteration += 1
    return {"iteration": self.actual_iteration - 1, "status": {}}


op.HKModel.iteration = _fixed_iteration
FixedHKModel = op.HKModel
