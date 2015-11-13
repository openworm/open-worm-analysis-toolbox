Motif analysis pipline
======================

(By Balázs Szigeti, OpenWorm contributor)

This document outlines my approach to behavioural analysis, the motif
analysis pipeline. It meant to complement the motion validation engine
developed by Michael and Jim. Neither methods are perfect in
characterising behaviour, hence using them in a complementary fashion
will enable to better understand how the behaviour of OpenWorm models
differ from that of real worms. I will keep updating this document as
progress happens, but feel free to add comments - especially if
something is not clear - or get in touch if you want to know more
details.

**What is motif analysis?**

The basic ideas behind the motif engine is that (1) behaviour is a
sequence of body shapes and that (2) body shapes can be represented in a
low dimensional vector space via the ‘eigenworm’ approach. Therefore
behaviour can be turned into a low dimensional time series and studied
with a wide array of time series analysis tools.

The first is step is to discretize behaviour, that is to look for
repeated patterns of body shapes. If a sequence of body shapes is
closely repeated over and over again, then it is safe to assume it is
not a random movement, but corresponds to a motor command. These are the
behavioural motifs/motor commands/states. Not surprisingly the results
of this analysis yields omega turns, runs, etc. In behavioural studies
typically these states are identified by some parameter exceeding a user
defined threshold. My analysis can also identify behavioural states, but
in an unsupervised manner.

Once the states are identified two basic types of analysis can be done:
Analyse the variability among the instances of the same behaviour (in a
similar fashion as the Brown paper ) Study the temporal relation among
the motifs. Most conveniently once behavioural states are identified,
behaviour can be easily converted to a Markov chain model

**What additional information this module will provide about
behaviour?**

The behavioural dictionary is working by extract features of behaviour.
These are like speed, frequency of omega turns etc. What is the key is
that most features are aggregated. Consider omega turns. Frequency of
omega turns is a feature, but it does not say when omega turns happen in
relation to other forms of behaviour.

For simplicity assume that behaviour consists only of turns and runs, T
and R respectively. Then the following worms have the same turn
frequency, although they are clearly different phenotypes by behaviour:

RRRTRRRT RRRTTRRR

Turn frequency is an aggregated feature - i.e. presented as a time
average - and therefore dependencies in the temporal domain are washed
out. Motif analysis preserves these correlations.Similarly the speed
feature is an aggregated measure, presented either as a mean value or a
distribution. However information can also reside in when the worm
speeds up.

Note that a simple Markov model could distinguish the phenotypes above.

**What is the status of the motif analysis?**

Bear in mind that it is also my PhD work. Strictly speaking the pipeline
is built up to analyse *Drosophila* larva (fruit fly) behaviour. However
because *Drosophila* larva and the worm share the basic body plan - i.e.
both of them are maggots -, my work can be applied to C. elegans with
minimal modification. I have obtained some data for the N2 worm from the
*C. elegans* behavioural database (CBD) and on this I have confirmed the
results of the eigenworm analysis.

The annotated videos updated to youtube by CBD are preprocessed and have
a colour coded contour. This is very annoying for me as my vision
algorithm was designed to work on simple videos. I am in the process of
acquiring the videos, but CBD is painfully slow to react to any of my
emails.

I have more *Drosophila* data from my own lab. As far as finding the
motifs there are surprisingly few algorithms for multidimensional motif
finding. Most methods are tested on either signals like heartbeat (that
are very-very stereotypical) or only give a very coarse grained motifs
(such as if the market is going up or down in economic time series).
However to make the motif engine meaningful I need more details than the
‘coarse grained economic description’, but my signal is much more
variable than heartbeats. I have found numerous methods that do not
work, but right now I am close to put together something that identifies
behaviours reliably. Currently I am using this method and the results
are promising for both Drosophila and the worm:
<http://epubs.siam.org/doi/abs/10.1137/1.9781611972825.77>
