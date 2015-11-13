Movement Validation White Paper
===============================

*(What follows is an excerpt from an unpublished revision (revised by
@MichaelCurrie) to "OpenWorm: An Open-Science Approach to Modelling
Caenorhabditis Elegans", by Balázs Szigeti, Padraig Gleeson, Matteo
Cantarelli, Michael Vella, Sergey Khayrulin, Andrey Palyanov, Michael
Currie, Jim Hokanson, Giovanni Idili, and Stephen Larson)*

Evaluating the worm simulation
------------------------------

Popper [39] reasoned that for a theory to be considered scientific, it
must make predictions that can in principle turn out to be false when
compared with observations. For example, general relativity predicts the
orbital period of Mercury. Astronomers can observe the orbital period of
Mercury and compare it to the value predicted by the theory. Without
falsifiable predictions, it is impossible to distinguish between
mutually incompatible theories of reality, which is the objective of a
scientific investigation.

In the domain of Artificial Intelligence (AI), Turing [40] predicted
that a legitimate simulated human intelligence could win an “imitation
game” (popularly known as the Turing Test). The simulation tries to fool
an interrogator chatting with it by text into thinking that the
interrogator was chatting with a real human. If the simulation fails to
fool the interrogator, the theory that it is a faithful reproduction of
a human intelligence should be considered false. By articulating a
falsifiable prediction of the theory that humans could be simulated on
computers, Turing rendered the notion of AI a scientific one.

A claim that a given simulated worm is a faithful reproduction of a real
worm can be rendered rigorous and scientific by a similar technique,
since one consequence, or prediction, of such a claim would be that the
worm could pass an analogous “imitation game” for worms. Of course,
worms can’t form sentences, so the interrogator of the worm imitation
game instead uses the contour outline of the worm body over time as the
test material. Harel [41] argued that if the simulated worm can
convincingly conduct itself as a real worm would in a variety of test
scenarios, it is has passed the Turing-like test.

A method to analyze the worm body contour over time has been already
developed by Schafer [42]. Schafer’s lab filmed hundreds of hours of
videos of wild-type and mutant worms grazing on bacterial lawns in Petri
dishes. The mutant worms were from 305 strains of *C. elegans* with
genes knocked down using RNAi. Since differences in behaviour between
wild-type and mutant worms can be too subtle to see by manual
observation [42], they further processed the worm videos into hundreds
of statistical “features”, such as speed of locomotion, head movement
amplitude, frequency of omega turns, etc. The features of a given mutant
were compared to a wild-type control. Differences in the genetics and
physiology of the mutant worms manifested as behavioural differences
that were statistically distinguishable in the calculated feature data.
That is, as desired, the mutant worms could not successfully “imitate”
the wild-type worms after their features were scrutinized. Furthermore,
this method enabled the lab to characterize the behavioural phenotype of
the mutant worms.

OpenWorm developed its “movement validation” tool by closely emulating
the Schafer Lab’s technique of automatically comparing worms based on
the quantifiable properties of their behaviour. The tool compares videos
of simulated and real worms by taking raw video, processing it into worm
contour data, and then further processing into features. The videos of
real worms are taken directly from the Schafer Lab’s dataset. A
generalized test process is outlined in figure 2.

![](test%20diagram.png)

Harel argued for a worm model verification framework that is "true to
all known facts". [18, 41] Since many facts are known about how the worm
behaves in response to stimuli, the Schafer lab dataset, which contains
only recordings of worms passively browsing in a bacterial layer, will
not suffice to fully test the model. Stimulus-response test scenarios
should be conducted, including: worms undergoing normal locomotion in
order to locate food or avoid a noxious stimulus, and worms stimulated
to move forward or backward by light touch.

Later, worm ethology should also be tested in richer physical
environments. Rather than merely simulating the worm on a plain Petri
dish, a three-dimensional soil environment like the natural habitat of
the worm could be used. Additionally, scenarios should be devised to
test appropriate changes to the behaviour of the worm over time. This
would extend the movement validation engine so the model could be tested
for its ability to reproduce the sleep-like states [43,44] and the
learning capabilities [45] of a real worm.

OpenWorm on every desk and in every home
----------------------------------------

The OpenWorm project has two goals. The first, discussed above, is to
functionally reproduce the behaviour of the wild-type *C. elegans* in a
variety of environmental contexts, to the extent that the simulated
behaviour is statistically indistinguishable from recordings of real
worms under analogous environmental conditions. The testing framework
discussed above should detect when this goal has been achieved. Note
that achieving this goal will answer the open question of what level of
detail the nervous system must be modelled at to preserve the emergent,
high-level behaviour of the worm.

OpenWorm’s second goal is for the simulation to be a faithful biological
model for *C. elegans*. Traditionally, AI research has attempted to
reproduce human-like intelligence without simulating the brain’s
physiology. OpenWorm could have taken a similar approach by attempting
an abstracted approach to modeling the behaviour of the worm without
modeling much underlying worm biology. Such an approach might very well
have been a quicker path to achieving the first goal. However, such a
model, even if it reproduced the macroscopic behaviour of the worm,
would provide limited scientific insight to biologists, who would be
unable to compare measurements they make in the lab to variables in the
simulation.

Instead, OpenWorm is being developed as a biological model, which will
make it straightforward to extract time series data for physiological
variables in the model, such as membrane potentials, ionic
concentrations, body wall forces, etc. Consequently, scientists can
modify model parameters, run the simulation, extract biological data,
and analyze the effects of the perturbation.

The achievement of this second goal will hopefully make OpenWorm an
indispensable software tool in *C. elegans* labs worldwide. Just like
modifications to cars are analyzed in computer-aided design (CAD)
programs before being tested on the road, scientists could make
perturbations *in silico* before beginning the expensive and
time-consuming work of conducting *in vivo* experiments. Conversely,
having many scientist users will engender a feedback process that will
make development more data-driven, helping to improve the biological
realism of the OpenWorm model in the first place.

References
----------

[18] David Harel. A grand challenge for computing: towards full reactive
modeling of a multi-cellular animal. Verification, Model Checking, and
Abstract Interpretation, pages 323–324. Springer, 2004.

[39] Karl R. Popper. The Logic of Scientific Discovery. Routledge, 1968.

[40] Alan M. Turing. Computing machinery and intelligence. Mind, pages
433–460, 1950.

[41] David Harel. A Turing-like test for biological modeling. Nature
Biotechnology, 23(4):495–496, 2005.

[42] Eviatar Yemini, Tadas Jucikas, Laura J. Grundy, André E. X. Brown,
and William R. Schafer. A database of *Caenorhabditis elegans*
behavioral phenotypes. Nature Methods, 10(9):877–879, 2013.

[43] David M. Raizen, John E. Zimmerman, Matthew H Maycock, Uyen D Ta,
Young-jai You, Meera V. Sundaram, and Allan I Pack. Lethargus is a
*Caenorhabditis elegans* sleep-like state. Nature, 451(7178):569–572,
2008.

[44] Julie Y. Cho and Paul W. Sternberg. Multilevel modulation of a
sensory motor circuit during *C. elegans* sleep and arousal. Cell,
156(1):249–260, 2014.

[45] Evan L. Ardiel and Catharine H. Rankin. An elegant mind: learning
and memory in *Caenorhabditis elegans*. Learning & Memory,
17(4):191–201, 2010.

Further Reading
---------------

[Ince et. al. The case for open computer programs. Nature Perspectives,
2012.](http://www.nature.com/nature/journal/v482/n7386/full/nature10836.html)

[Cyrus Omar, Jonathan Aldrich, Richard C. Gerkin. Collaborative
Infrastructure for Test-Driven Scientific Model
Valuation.](https://github.com/cyrus-/papers/raw/master/sciunit-icse14/sciunit-icse14.pdf)
([SciUnit](https://github.com/scidash/sciunit)). 36th International
Conference on Software Engineering, Hyderabad, India, 2014.
