To: Daniel Arias-Aranda <darias@go.ugr.es>, Peyman Rabiei <prabiei@correo.ugr.es>
Cc: Vladimir Stantchev <vladimir@stantchev.com>
Subject: Systems paper: major revision, for your approval
Attachments: paperB-mdpi-revised.pdf, paperB-mdpi-diff-highlighted.pdf, response-reviewer-1.pdf, response-reviewer-2.pdf, response-reviewer-3.pdf

Dear Daniel, dear Peyman,

The *Systems* paper (systems-4669426) came back on 3 November as a major revision, with three reports. The revised version is due on 13 November. Attached are the revised manuscript, a version with all changes highlighted, and one response letter per reviewer.

What changed:

1. New title: "Is the Optimiser Needed? Auditing a Digital Decision-Support System for Necessity, Decision Stability and Input Fragility under Turbulence". The reviewers asked us to separate the stochastic optimiser from the fuzzy decision support, and to stop presenting the paper as evidence about economic turbulence. The economic link is now an analogy of mechanism only.
2. Steps S1–S3 of the protocol now also run on two further models: Peyman's ED volunteer-staffing model, and the multi-objective knapsack benchmark of Zitzler and Thiele. On the knapsack, the evolutionary search is close to optimal and stable, and the paper says so.
3. "Identifiability" is now "decision stability". The paper shows that the solver stops where the model admits many near-equivalent allocations, and that an exact decider with a stated tie-breaking rule commits to one allocation.
4. A new Section 3.2 defines the model, so the paper no longer relies on the *Applied Sciences* article for its definitions.
5. The deployment envelope was rerun with stronger baselines and full timing. Beyond the sizes where the exact method proves optimality, a capacity-aware local search works best under tight time limits.

While doing the revision I found two problems in our own work. The paper reports both:

- The instance generator for the large scaling instances was not reproducible across runs, so two rows of the old Table 6 had been computed on different instances. This is fixed, and the envelope has been rerun on stored instances.
- The structural check also found that the ULPP objective, as written, prefers to direct the people of lowest priority. FIS1 gives the highest ULPP to the most vulnerable people (high vulnerability, severe damage, little resource time), and the objective minimises mean ULPP over the people who are directed. On the reference instances, the exact optimum directs only 12% (small) and 7% (large) of the people the crisp priority rule ranks highest.

  I know how this happened. Early versions of the *Applied Sciences* paper worked with FLPP, the fairness of leaving the vulnerable behind. We then changed to ULPP, the unfairness of leaving someone behind. I did not update all the references and formulas that depended on the direction of the score, and two or three of them slipped through our own reviews and the peer review: the objective still sums the score over the people directed instead of those left out, the text still reads a low ULPP as "directing the most vulnerable", and one sentence still says "FLPP". The rule base and the definition of ULPP are right; the objective runs over the wrong set of people.

  The *Systems* paper reports this as a finding of the structural check and names the cause, an objective that runs over the wrong set (Section 5.1). Its results concern the model as published and do not change. For the *Applied Sciences* article it is a correction to the objective, not only to the text, and the results that depend on which people are directed will change. I will add it to the correction we are preparing for that article and send you the draft separately.

  Peyman, please confirm that this matches the model as you built it: ULPP_j is the unfairness of leaving person j out.

What I need from each of you is a short written reply confirming that you approve the revised version, specifically:

- the new title and abstract;
- the ULPP finding and the paragraphs on the *Applied Sciences* article (Sections 5.1 and 6.5 and the Conclusions), including the statement that a correction to that article is in preparation;
- the disclosure of the instance-generator defect;
- the three response letters;
- the author order, the author contributions and the AI-use statement, all unchanged.

The Data Availability statement will cite a new Zenodo version of the code (0.5.0) once I have released it. That is the only change still to come. I plan to upload on 12 November, so a reply by 11 November would keep a day in reserve.

Best regards,
Vladimir
