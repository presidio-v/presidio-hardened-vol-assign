Dear Ms. Chang, dear Guest Editors Prof. Jong-min Kim and Prof. Rob Kim Marjerison,

We submit the revised version of manuscript systems-4669426 for the Special Issue "Using Digital AI Systems as a Response to High Economic Turbulence and Uncertainty". We thank the three reviewers for their reports. Each point is answered in a separate letter per reviewer, and all changes are marked in the version with changes highlighted.

The revision changes what the paper claims as well as how it says it:

- **Title and scope.** The new title is "Is the Optimiser Needed? Auditing a Digital Decision-Support System for Necessity, Decision Stability and Input Fragility under Turbulence". As the reviewers asked, the paper now separates the stochastic optimiser, which the experiments show to be unnecessary, from the fuzzy decision support, which they do not test. The economic setting is engaged as an analogy of mechanism only, and no quantitative result is presented as a statement about commercial systems.
- **Further cases.** Steps S1–S3 of the protocol now also run on two further models: an ED volunteer-staffing model and a multi-objective knapsack benchmark that is not ours. The protocol reaches a different verdict on each, including one where the evolutionary search is close to optimal and stable.
- **Decision stability.** "Identifiability" is replaced by "decision stability". The paper measures the multiplicity of near-optimal allocations directly and demonstrates a tie-breaking rule under which an exact decider commits to one allocation.
- **Methods and evidence.** A new section defines the model in full. The turbulence study now covers 36 conditions per size with joint perturbations and systematic bias, the deployment envelope compares stronger baselines on stored instances with full timing, and the paper is positioned against algorithmic-auditing and AI-governance frameworks.

While revising, we found two problems in our own work, and the paper reports both. The generator of the scaling instances was not reproducible across runs, so two rows of a table in the submitted version had been computed on different instances; this is fixed and the affected results were rerun. The structural check also showed that the prioritisation objective of our earlier Applied Sciences article (doi:10.3390/app16157581), as written, sums its unfairness score over the wrong set of people. A correction to that article is in preparation.

As requested in your message of 6 November, the manuscript now uses the updated template and includes the Highlights section. We also provide a graphical abstract. The code, instances and result manifests of this revision are archived at Zenodo (https://doi.org/10.5281/zenodo.24057413).

All authors have approved the revised version.

Sincerely,

Vladimir Stantchev (corresponding author), on behalf of all authors
Institute of Information Systems, SRH University Heidelberg, Germany
stantchev@computer.org
