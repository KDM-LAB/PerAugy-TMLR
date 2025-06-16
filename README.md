Steps:
1. Download PENS dataset : It contains train set, validation set, test set and a news set.
2. Run PENS_augmentation.py in Scripts : This will arrange the pens dataset into seed UIG format. We order the interactions as per timestep of each user, and then append summary nodes from the test set. The seed UIG is created and saved.  
3. Save the synthetic-original.csv dataset : This is the seed UIG for pens. Also we get a corresponding summary dataset which has all the informations about summary nodes created by each synthetic user in seed trajectory pool.
4. Run DS/DoubleShuffling.ipynb on it : This will create the Double Shuffled trajectories. You can play around with offset, gap, segment length and other settings. 
5. Run perturbation/perturbation_D2.ipynb : After double shuffling, this will smoothen the DS dataset by applying history influenced perturbation. 
6. Convert to PENS format by KG2PENS/KG2PENS_Trainer_New_Convertor.ipynb : We replace the <d-s> pairs in the dataset with s-nodes only and then convert the dataset into pens KG format for sequential recommendation task. The final dataset is of format USerID, ClicknewsID, pos, neg.  
   
