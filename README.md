## Overview
Hidden Markov Models are used extensively in Artificial Intelligence, Pattern Recognition, Computer Vision, and many other fields.  If a system has unobservable (hidden) states and each state is independent of the prior, then we can create a model of that system using probability distributions over a sequence of observations.  The idea is that we can provide this system with a series of observations to use to query what is the most likely sequence of states that generated these observations.

### Local Testing
If you are using `submission.py` to complete the assignment instead of the Jupyter Notebook, you can run the tests using:

`python hmm_submission_tests.py`

This will run all unit tests for the assignment, comment out the ones that aren't related to your part (at the bottom of the file) if going step by step.

## The Assignment
The goal of this assignment is to demonstrate the power of probabalistic models. You will build a word recognizer for American Sign Language (ASL) video sequences. In particular, this project employs [hidden Markov models (HMM's)](https://en.wikipedia.org/wiki/Hidden_Markov_model) to analyze a series of measurements taken from videos of American Sign Language (ASL) collected for research (see the [RWTH-BOSTON-104 Database](http://www-i6.informatik.rwth-aachen.de/~dreuw/database-rwth-boston-104.php)).

In each video, an ASL signer is signing a meaningful sentence. In a typical ASL recognition system, you observe the XY coordinates of the speaker's left hand, right hand, and nose for every frame. The following diagram shows how the positions of the left hand (Red), right hand (Blue), and nose (Green) change over time in video number #66. Saturation of colors represents time elapsed.

<img src="./demo/hands_nose_position.png" alt="hands nose position">

In this assignment, for the sake of simplicity, you will only use the Y-coordinates of each hand to construct your HMM. In Part 1 you will build a one dimensional model, recognizing words based only on a series of right-hand Y coordinates; in Part 2 you will go multidimensional and utilize both hands. At this point, you will have two observed coordinates at each time step (frame) representing right hand & left hand Y positions.

The words you will be recognizing are "BUY", "HOUSE", and "CAR". These individual signs can be seen in the sign phrases from our dataset:

<img src="./demo/buy_house_slow.gif"> 

<p style="text-align:center; font-weight:bold"> JOHN CAN BUY HOUSE </p> 

<img src="./demo/buy_car_slow.gif"> 

<p style="text-align:center;  font-weight:bold"> JOHN BUY CAR [FUTURE] </p> 


### Part 1a: Encoding the HMM

Determine following values for each word:
1. the transition probabilities of each state
2. the mean & standard deviation of emission Gaussian distribution of each state

Use the training samples from the table below. Provide the transition and prior probabilities as well as the emission parameters for all three words with **accuracy to 3 decimal digits**.

Round the values to 3 decimal places thoughout entire assignment:
- 0.1 stays 0.1 or 0.100
- 0.1234 rounds to 0.123
- 0.2345 rounds to 0.235
- 0.3456 rounds to 0.346
- 0.0123 rounds to 0.012
- 0.0125 rounds to 0.013

Those values can be hardcoded in your program. Don't use round() from python.

Word | Frames | Observed sequence | Initial State1 | Initial State2 | Initial State3
--- | --- | --- | --- | --- | --- 
BUY | 6 | 36, 44, 52, 56, 49, 44 | 36, 44 | 52, 56 | 49, 44
BUY | 8 | 42, 46, 54, 62, 68, 65, 60, 56 | 42, 46, 54 | 62, 68, 65 | 60, 56
BUY | 10 | 42, 40, 41, 43, 52, 55, 59, 60, 55, 47 | 42, 40, 41|43, 52, 55|59, 60, 55, 47
CAR | 10 | 47, 39, 32, 34, 36, 42, 42, 42, 34, 25|47, 39, 32|34, 36, 42|42, 42, 34, 25
CAR | 9 | 35, 35, 43, 46, 52, 52, 56, 49, 45|35, 35, 43|46, 52, 52|56, 49, 45
CAR | 8 | 28, 35, 46, 46, 48, 43, 43, 40|28, 35, 46|46, 48, 43|43, 40
HOUSE| 15 | 37, 36, 32, 26, 26, 25, 23, 22, 21, 39, 48, 60, 70, 74, 77|37, 36, 32, 26, 26|25, 23, 22, 21, 39|48, 60, 70, 74, 77
HOUSE| 15 | 50, 50, 49, 47, 39, 39, 38, 38, 50, 56, 61, 67, 67, 67, 67|50, 50, 49, 47, 39|39, 38, 38, 50, 56|61, 67, 67, 67, 67
HOUSE| 16 | 45, 43, 44, 43, 40, 35, 36, 37, 39, 45, 60, 68, 66, 72, 72, 75|45, 43, 44, 43, 40|35, 36, 37, 39, 45|60, 68, 66, 72, 72, 75

As shown in the diagram below, each one of the three words (BUY, CAR, and HOUSE) has exactly **THREE hidden states** in its HMM. All words must start from State 1 and can only transit to the next state or stay in the current one.

<img src="part_1_a_probs.png">

### _Training sequences need to have 3 hidden states no matter what!_
If you follow the HMM training procedure described in Canvas, you might encounter a situation where a hidden state is **_squeezed_** out by an adjacent state; that is, a state might have its only observation moved to another state. In that situation, always keep at least one observation for that hidden state.

Example:
Assume you've reached a stage where the following is true: 
- State 1 has mean=53 & std=6
- State 2 has mean=37 & std=9
- State 3 has mean=70 & std=8

The next training sample has the following observed sequence:

`45 45 34 | 30 30 25 36 52 | 62 69 74` 

and you are trying to adjust the location of state boundary between State 1 & 2. You first move it 1 step to the left since 34 is closer to State 2, and then you realize that 45 is still closer to State 2. If you follow the same routine, you will end up with no obvervation for State 1. In order to prevent this from happening, you have to stop at the last "45" and as a result leave the boundary as 

`45 | 45 34 30 30 25 36 52 | 62 69 74`

Now you meet the '3 hidden states per sample' requirement.

### Some hints/guidelines for training
#### How should we compare if an observation if closer to one state or another?
Check how many standard deviations away is the observation from the mean for each state. 
Example: Say 46 is the rightmost observation in S1. If we denote the mean and std of state i as μi,σi, then should we be comparing 
|46−μ1| / σ1 vs |46−μ2| / σ2

#### For HMM training, which side of the boundary should we check first while assigning observed sequence values to states?
After computing the mean and std for each state, adjust the boundary between the states. Always start from the 1st element at the LEFT side of the boundary. If the LEFT element is closer to the next state, then move the boundary leftward. If the LEFT element should stay at the current state, then check the RIGHT element. This is just done to make sure that everyone gets the same results in the context of the assignment.

---

### Part 1b: Creating the Viterbi Trellis

The goal here will be to use the HMM derived from Part 1a (states, prior probabilities, transition probabilities, and parameters of emission distribution) to build a Viterbi trellis.  When provided with an evidence vector (list of observed right-hand Y coordinates), the function will return the most likely sequence of states that generated the evidence and the probabilty of that sequence being correct.

For example, an evidence vector [36, 44, 52, 53, 49, 44] should output a sequence ['B1', ... 'B2', ... 'B3']

If no sequence can be found, the algorithm should return one of the following tuples:
`(None, 0)` (null),  `([], 0)` (empty list) or  `(['C1', 'C1', ... 'C1'],0)` (Or all being the first state of that letter)

"No sequence can be found" means the probability reaches 0 midway. If you find an incomplete sequence with some probability, output that sequence with its probability. 


#### Hint:
In order to reconstruct your most-likely path after running Viterbi, you'll need to keep track of a back-pointer at each state, which directs you to that state's most-likely predecessor.

You are asked to use the provided function `gaussian_prob` to compute  emission probabilities. In a typical HMM model you have to convert the probability to log-base in order to prevent numerical underflow, but in this assignemnt we will only test your function against a rather short sequence of observations, so **DO NOT** convert the probability to logarithmic probability or you will fail on Gradescope.

#### Gradescope:
In the autograder, we will also test your code against other `evidence_vectors`.

----

### Part2a: Multidimensional Output Probabilities

In Part 1a, we use only right-hand Y-axis coordinates as our feature, and now we are going to use both hands. Since ASL is two handed, using observations from both the right and left hands as features can increase the accuracy of our model when dealing with more complex sentences.

Here you are given the transition probabilities and the emission parameters of left-hand Y-axis locations, following the same procedure conducted in Part 1a.

One thing to notice is, in Part 1, the `viterbi` function is tested against single words. That is, the input evidence vector will not transit between different words. However, for Part 2, the input evidence vector can be either a single word, or a verb phrase such as "BUY CAR" and "BUY HOUSE". Adjust the given transition probabilities to adapt to this fact.

*NOTE: Add NEW keys to the transition dictionary ONLY if there is a NON-ZERO transition probability*

<img src="part_2_a_probs.png" alt="2a_probs">

BUY | State 1 | State 2 | State 3
--- | --- | --- | --- 
Mean | 108.200 | 78.670 | 64.182
Std | 17.314 | 1.886 | 5.573

CAR | State 1 | State 2 | State 3
--- | --- | --- | --- 
Mean | 56.300 | 37.110 | 50.000
Std | 10.659 | 4.306 | 7.826

HOUSE | State 1 | State 2 | State 3
--- | --- | --- | --- 
Mean | 53.600 | 37.168 | 74.176
Std | 7.392 | 8.875 | 8.347

---

### Part 2b: Improving the Viterbi Trellis

Modify the Viterbi trellis function to allow multiple observed values (Y location of right and left hands) for a state. The return format should be identical to Part 1b.
