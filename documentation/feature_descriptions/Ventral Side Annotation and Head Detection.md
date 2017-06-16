Ventral Side Annotation and Head Detection
==========================================

From From Yemini, E., Jucikas, T., Grundy, L. J., Brown, A. E. X. &
Schafer, W. R. A database of Caenorhabditis elegans behavioral
phenotypes. Nat. Methods (2013). `doi:10.1038/nmeth.2560`

Our worm features necessitate dorsal-ventral and head-tail distinctions.
The worm's ventral side was annotated for each video by eye. We did not
profile rolling mutants and therefore expected worms to maintain their
dorsal-ventral orientation. Nevertheless, 126 random videos were
examined and the worms therein found never to flip sides. Head-tail
orientation was annotated automatically by software. We examined 133
random videos (roughly lt of our data, 2.25 million segmented frames), a
collection of 100 from a quality-filtered set and 33 rejected by this
filter (see the section titled "Feature File Overview"), representing a
broad range of mutants !including several nearly motionless UNCsl. Many
of these include early videos which suffered multiple dropped frames and
poor imaging conditions that were later improved. We found that the head
wascorrectly labeled with a mean and standard deviation of 94.39 ±
17.54% across individual videos and 95.6% of the video frames
collectively.

Before assigning the head and tail, videos are split into chunks in
which worm skeletons can be confidently oriented with respect to each
other. Chunk boundaries are set whenever there is a gap in skeletonized
frames of 0.25 seconds or more. During these gaps, worm motion could
make skeleton orientation unreliable. The skeletons within each chunk
are aligned by determining which ofthe two possible head-tail
orientations minimizes the distance between corresponding skeleton
points in subsequent frames. When possible, we unify chunks and heal up
to 1/2 second interruptions by determining whether the worm was bent
enough to achieve an omega turn and flip its orientation. If so, we
trace the worm's path through its large bend to determine the new
orientation. If the path cannot be confidently traced, we avoid healing
and maintain separate chunks.

The head is detected in each chunk of oriented frames. The head and neck
perform more lateral motion (e.g., foraging) even in uncoordinated
mutants. Therefore, we measure lateral motion at both worm endpoints,
across each chunk - unless the chunk is shorter than 1/6 of a second
which is too short to reliably measure such notion. In our setup, the
head is lighter than the tail. Therefore, we also measure the grayscale
intensity at both worm endpoints, across each chunk. Linear discriminant
analysis (LDA) was used on a combination of lateral notion and intensity
at the worm endpoints for a training set of 68 randomly-chosen videos.
This classifier was then used for the entire data set to automatically
detect and label the worm's head.
