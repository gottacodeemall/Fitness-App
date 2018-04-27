# Fitness-App

**Abstract**
The app overlays the label of the exercise the person is performing using human pose estimation in real time.

**Procedure**
The video from the feed is fed frame by frame and the joint locations of the person in the video are extracted using the code written by the researchers at Carnegie Mellon university and then the joint location data is fed to a support vector machine trained using custom features generated from the joint location data. The video is then stitched together with the overlayed label of the exercise.
We achieved a speed of 1 frame/second using the above mentioned method by running it on a CPU.

**Fun video**
https://www.youtube.com/watch?v=SojZ0j6nH0k 
