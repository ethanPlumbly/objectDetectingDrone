# objectDetectingDrone
Autonomous drone research has been a key area of innovation in 
computer vision in recent years. This project details the 
research and implementation of an autonomous Tello drone system 
that extracts and displays a live camera feed that is then processed 
using machine learning techniques through TensorFlow. Open palm 
gestures are localised and classified within each frame and displayed 
using bounding box overlays. The coordinates for these boxes are 
then used to determine the boxes relative position within the image 
such that orthogonal velocities can be calculated that will centre the 
gesture on the screen. Once these values are calculated, the velocities 
are transmitted back to the done using an interface library resulting in 
corresponding real-world movement. This results in an autonomous 
system that allows the user to move the drone without directly 
controlling it.

In total, three TensorFlow network architectures are trained using a 
custom data set that I collected and formatted myself. Example images can be found in the training_image_examples folder. 
This dataset was collected within a single room under varying lighting conditions to 
diversify the data. The drone was suspended from varying positions 
and several angles of the open palm gesture were recorded to ensure 
as many scenarios are represented as possible. The data is then 
labelled using LabelImg software and converted to TFRecords to be 
trained by the aforementioned TensorFlow models. Models are 
specifically chosen to cover a diverse range of accuracies and speed 
attributes. The lightweight MobileNet priorities speed over accuracy, 
the middleweight R-CNN balances accuracy and speed equally, and 
the heavyweight RetinaNet prioritises accuracy over speed.

Once training is complete for each network (or ended early for the 
RetinaNet due to taking too long), a vigorous evaluation process has been 
undertaken that analyses each network’s training through 
TensorBoard, TensorFlow’s visualisation toolkit. This evaluation can be found in the project report
This allowed several properties of the training process to be studied such as localisation 
loss, classification loss and learning rate. Practical tests are also 
performed that showcase various scenarios that the drone might 
encounter as well as some possible visual glitches that occur due to 
misconfiguration. Surprisingly, the RetinaNet ended up performing the 
worst while the middleweight Faster R-CNN achieved the highest 
accuracy scores. Unfortunately, due to the longer processing speed, 
the R-CNN is deemed unsuitable for processing live video, so the still 
relatively efficient MobileNet is left as the only appropriate architecture. 
