# Plans for improvements

## Image Preprocessing
Mediapipe can be inconsistent with keeping track of the hands, so some image preprocessing may help. Here's some ideas we can try:

- Improve contrast
- Add gaussian blur to reduce noise

## Landmarks Preprocessing
- Normalize coordinates by position (it shouldn't matter what part of the frame the hands are)
- Normalize by scale (to cover larger/smaller hands or distance from camera)
- Normalize by rotation
  - Ideally we would normalize by every axis, but from what I've seen, the Z coordinate created from the landmarks, is inferred and can be inaccurate.
  So I think normalizing by just the X-Y plane would be best.
  For example: we could rotate the coordinates such that the thumb-side wrist point is always (0,0) and the largest pinky knuckle is has X=0. Or something like that
  - I imagine we could use matrix multiplication to do all these translations

## Metrics
Since one of the requirements for the report is to measure results, I thought of a way to get a metric for the program:

- Take some subset of the Mudras
- Make a program that cycles through them with a 3 second counter in between
- During some time frame (like 10 seconds), it will show the desired Mudra, and one of us will make that gesture in many different positions
- For each frame that it classifies it correctly, it will increase a tally, and then the success rate will be that tally over total frames.
- It will then cycle through a few of the Mudras automatically, and save the success rates for those classifications.
- I hesitate to use all of the Mudras in the training set, just because I don't know how to do some of them.