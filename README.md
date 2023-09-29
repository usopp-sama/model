# model
 open-cv and yolo implementation



first commit for arrow detection, it works on the .h5 file taken from teachable machines.


keras_model.h5 : h5 file after a "better dataset", but has fucked up left and right arrow detect. and environment detection with a limited dataset

keras_model_x.h5: prev h5 file

labels.txt: text file corresponding to keras_model

labels_x.txt: text file corresponding to keras_model_x

my.py : opencv code for arrow detdection

ros.py : yet to be implemented, but it initiates ros nodes as subscriber and listener to print camera output onto /rosout, when echoed.



chat-gpt:
https://chat.openai.com/share/f8092b93-c6c9-4559-902e-9c1e3f5ddf5c (arrow detection chat)


https://chat.openai.com/share/903a56d2-2d32-4c94-8ed4-7331c894d513 (ros integration chat)
