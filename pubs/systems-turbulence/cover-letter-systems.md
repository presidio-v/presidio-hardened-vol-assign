Dear Ms. Zhao, dear Guest Editors Prof. Jong-min Kim and Prof. Rob Kim Marjerison,

We submit the manuscript "Is the AI Needed? Auditing a Digital Decision System for Necessity,
Identifiability and Input Fragility under Turbulence" for the Special Issue "Using Digital AI
Systems as a Response to High Economic Turbulence and Uncertainty" of *Systems*.

The paper audits a digital AI decision system, and the system is our own. It is a fuzzy
multi-objective allocation tool for post-disaster relief that we published with an open
reference implementation in *Applied Sciences* (16(15):7581, doi:10.3390/app16157581). The
audit shows that the published optimisation model is solved exactly by a deterministic rule
that dominates the evolutionary search, that the system's committed decision changes for most
of the people it directs between two equally valid runs, and that its true input fragility only
becomes visible once solver noise is removed. We turn these steps into a five-step audit
protocol that any organisation can run before trusting an optimising AI system in turbulent
conditions.

Scope. The case is relief allocation after a disaster, an acute resource-allocation shock. We
engage the Special Issue's economic-turbulence theme through an explicit mapping of shock
mechanisms (demand exceeding capacity, disrupted logistics, degraded and late data) rather than
with market data, and we state this boundary in the paper. The deliverable for the Special
Issue's readers is the protocol, which does not depend on the domain.

Relation to our earlier article. This manuscript is not a second report of the same results.
The *Applied Sciences* article proposes the model and compares evolutionary algorithms with
each other. The present manuscript tests whether those algorithms are needed at all, measures
the identifiability of their output, re-measures input fragility on an exact decider, and
repairs a semantic gap in the published model. All analyses, code and data are new; the
earlier article is cited wherever its model or instances are used. \[PENDING: co-author
decision on a correction to the earlier article — mention here if filed.\]

The data, code and result manifests are openly available (\[PENDING: Zenodo DOI\]). The
manuscript has not been published or submitted elsewhere, and all authors have approved the
submission. As agreed with Ms. Zhao on 7 July 2026, we would be grateful if the full APC waiver
offered for this Special Issue could be applied once the manuscript is sent for review.

Sincerely,

Vladimir Stantchev (corresponding author), on behalf of all authors
Institute of Information Systems, SRH University Heidelberg, Germany
stantchev@computer.org
