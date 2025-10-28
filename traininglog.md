28/10/2025 09:44:09 Epoch: 0019, Train Loss: 29.16127 [Label: 1.099, Struct: 24.8
89, Feat: 1.507, Hom: 1.132, KL: 3.131], Val Loss: 28.69107                      
-------------------------------------------------------------------------------- 
                                                                                 
✓ Saved best autoencoder checkpoint (val_loss=28.69107) [NEW BEST!]              
[Epoch 20/20] Training phase started...                                          
  Batch [10/175] - Loss: 29.12332, GPU mem: 0.07/0.79 GB                         
  Batch [20/175] - Loss: 29.44307, GPU mem: 0.07/0.79 GB                         
  Batch [30/175] - Loss: 29.54590, GPU mem: 0.07/0.79 GB                         
  Batch [40/175] - Loss: 29.79522, GPU mem: 0.07/0.79 GB                         
  Batch [50/175] - Loss: 29.56006, GPU mem: 0.07/0.79 GB                         
  Batch [60/175] - Loss: 28.62688, GPU mem: 0.07/0.79 GB                         
  Batch [70/175] - Loss: 29.59154, GPU mem: 0.07/0.79 GB                         
  Batch [80/175] - Loss: 28.96230, GPU mem: 0.07/0.79 GB                         
  Batch [90/175] - Loss: 28.28154, GPU mem: 0.07/0.79 GB                         
  Batch [100/175] - Loss: 28.70076, GPU mem: 0.07/0.79 GB                        
  Batch [110/175] - Loss: 29.55864, GPU mem: 0.07/0.79 GB                        
  Batch [120/175] - Loss: 29.54563, GPU mem: 0.07/0.79 GB                        
  Batch [130/175] - Loss: 28.64597, GPU mem: 0.07/0.79 GB                        
  Batch [140/175] - Loss: 29.03393, GPU mem: 0.07/0.79 GB                        
  Batch [150/175] - Loss: 29.48654, GPU mem: 0.07/0.79 GB                        
  Batch [160/175] - Loss: 29.49124, GPU mem: 0.07/0.79 GB                        
  Batch [170/175] - Loss: 29.61734, GPU mem: 0.07/0.79 GB                        
[Epoch 20/20] Training complete. Starting validation...                          
[Epoch 20/20] Validation complete. Time: 37.9s                                   
                                                                                 
-------------------------------------------------------------------------------- 
28/10/2025 09:44:47 Epoch: 0020, Train Loss: 29.09720 [Label: 1.099, Struct: 24.8
32, Feat: 1.495, Hom: 1.132, KL: 3.092], Val Loss: 28.73609                      
-------------------------------------------------------------------------------- 
                                                                                 
  Val loss (28.73609) did not improve from best (28.69107)                       
                                                                                 
============================================================                     
Autoencoder Training Complete!                                                   
Best validation loss: 28.69107                                                   
============================================================                     
                                                                                 
                                                                                 
================================================================================ 
Evaluating Autoencoder Performance                                               
================================================================================ 
Test set size: 22 batches                                                        
Computing graph similarity metrics (Weisfeiler-Lehman kernel)...                 
================================================================================ 
                                                                                 
⚠ Warning: Could not compute WL kernel for some graphs: Graph does not have any l
abels for vertices.                                                              
  This may happen if some reconstructed graphs are empty or malformed.           
⚠ Warning: Could not compute WL kernel for some graphs: Graph does not have any l
abels for vertices.                                                              
  This may happen if some reconstructed graphs are empty or malformed.           
⚠ Warning: Could not compute WL kernel for some graphs: Graph does not have any l
abels for vertices.                                                              
  This may happen if some reconstructed graphs are empty or malformed.           
⚠ Warning: Could not compute WL kernel for some graphs: Graph does not have any l
abels for vertices.                                                              
  This may happen if some reconstructed graphs are empty or malformed.           
⚠ Warning: Could not compute WL kernel for some graphs: Graph does not have any l
abels for vertices.
