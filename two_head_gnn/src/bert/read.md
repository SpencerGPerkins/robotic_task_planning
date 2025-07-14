gai_action:
0: holding
1: on
2: insert
3: lock
4: want faster
5: want slower
6: want stop
7: without wire
8: without terminal

gai_wire:
0: red
1: blue
2: yellow
3: green
4: black

gai_terminal:
same as the terminal num

-------- above files can train bert by using train_debert.py

gai_merged_wire_terminal:
combine wire and terminal, it can output a vector that embed wire and terminal

--------

dim of the data and train file should be matched ex: train on action, num_classes = 9