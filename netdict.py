"""
Created on Thu Dec 12 15:45:07 2019

@author: stormlab
"""

#dictionary with the different neural network structures tested
def Net_type(dropout, reg, kernel_init):
    net_kernel_params = {
            'cpu_net': {            
                'down_conv_kernels': [
                    [(5, 128, 0, (0,0), kernel_init), (5, 128, dropout, (0,0), kernel_init)],  
                    [(5, 256, 0, (0,0), kernel_init), (5, 256, dropout, reg, kernel_init)],
                    [(5, 256, 0, (0,0), kernel_init), (5, 256, dropout, (0,0), kernel_init)],
                    [(5, 512, 0, (0,0), kernel_init), (5, 512, dropout, reg, kernel_init)],
                ],
                'lstm_kernels': [
                    [(5, 128, dropout, (0,0), kernel_init)], 
                    [(5, 256, dropout, reg, kernel_init)],
                    [(5, 256, dropout, (0,0), kernel_init)],
                    [(5, 512, dropout, reg, kernel_init)],
                ],
                'up_conv_kernels': [
                    [(5, 256, 0, (0,0), kernel_init), (5, 256, dropout, (0,0), kernel_init)],
                    [(5, 128, 0, (0,0), kernel_init), (5, 128, dropout, reg, kernel_init)],
                    [(5, 64, 0, (0,0), kernel_init), (5, 64, dropout, (0,0), kernel_init)],
                    [(5, 32, 0, (0,0), kernel_init), (5, 32, dropout, reg, kernel_init), (1, 1, dropout, reg, kernel_init)],
                ],
            },   
            'imagenet_net': {            
                'down_conv_kernels': [
                    [(3, 64, 0, (0,0), kernel_init), (3, 64, dropout, (0,0), kernel_init)],  
                    [(3, 128, 0, (0,0), kernel_init), (3, 128, dropout, reg, kernel_init)],
                    [(3, 256, 0, (0,0), kernel_init), (3, 256, dropout, (0,0), kernel_init), (5, 256, dropout, (0,0), kernel_init)],
                    [(3, 512, 0, (0,0), kernel_init), (3, 512, dropout, reg, kernel_init), (5, 512, dropout, (0,0), kernel_init)],
                ],
                'lstm_kernels': [
                    [(3, 128, dropout, (0,0), kernel_init)], 
                    [(3, 256, dropout, reg, kernel_init)],
                    [(3, 256, dropout, (0,0), kernel_init)],
                    [(3, 512, dropout, reg, kernel_init)],
                ],
                'up_conv_kernels': [
                    [(3, 256, 0, (0,0), kernel_init), (3, 256, dropout, (0,0), kernel_init)],
                    [(3, 128, 0, (0,0), kernel_init), (3, 128, dropout, reg, kernel_init)],
                    [(3, 64, 0, (0,0), kernel_init), (3, 64, dropout, (0,0), kernel_init)],
                    [(3, 32, 0, (0,0), kernel_init), (3, 32, dropout, reg, kernel_init), (1, 1, dropout, reg, kernel_init)],
                ],
            },   
            'deeper_net': {
                'down_conv_kernels': [
                    [(5, 32, 0, (0,0), kernel_init), (5, 32, dropout, (0,0), kernel_init)],
                    [(5, 64, 0, (0,0), kernel_init), (5, 64, dropout, (0,0), kernel_init)],
                    [(5, 64, 0, (0,0), kernel_init), (5, 64, dropout, reg, kernel_init)],
                    [(5, 128, 0 , (0,0), kernel_init), (5, 128, dropout, (0,0), kernel_init)],
                    [(5, 256, 0, (0,0), kernel_init), (5, 256, dropout, reg, kernel_init)],
                ],
                'lstm_kernels': [
                    [(5, 32, dropout, (0,0), kernel_init)],
                    [(5, 64, dropout, reg, kernel_init)],
                    [(5, 64, dropout, (0,0), kernel_init)],
                    [(5, 128, dropout, (0,0), kernel_init)],
                    [(5, 256, dropout, reg, kernel_init)],                
                ],
                'up_conv_kernels': [
                    [(5, 128, 0, (0,0), kernel_init), (5, 128, dropout, (0,0), kernel_init)],
                    [(5, 64, 0, (0,0), kernel_init), (5, 64, dropout, (0,0), kernel_init)],
                    [(5, 32, 0, (0,0), kernel_init), (5, 32, dropout, reg, kernel_init)],
                    [(5, 32, 0, (0,0), kernel_init), (5, 32, dropout, (0,0), kernel_init)],
                    [(5, 16, 0, (0,0), kernel_init), (5, 16, dropout, reg, kernel_init), (1, 1, dropout, reg, kernel_init)],
                ],
            },
            'original_net': {
                'down_conv_kernels': [
                    [(5, 64, 0, (0,0), kernel_init), (5, 64, dropout, (0,0), kernel_init)],
                    [(5, 128, 0, (0,0), kernel_init), (5, 128, dropout, reg, kernel_init)],
                    [(5, 128, 0, (0,0), kernel_init), (5, 128, dropout, (0,0), kernel_init)],
                    [(5, 256, 0, (0,0), kernel_init), (5, 256, dropout, reg, kernel_init)],
                ],
                'lstm_kernels': [
                    [(5, 64, 0, (0,0), kernel_init)],
                    [(5, 128, 0, reg, kernel_init)],
                    [(5, 128, 0, (0,0), kernel_init)],
                    [(5, 256, dropout, reg, kernel_init)],
                ],
                'up_conv_kernels': [
                    [(5, 256, 0, (0,0), kernel_init), (5, 256, dropout, (0,0), kernel_init)],
                    [(5, 128, 0, (0,0), kernel_init), (5, 128, dropout, reg, kernel_init)],
                    [(5, 128, 0, (0,0), kernel_init), (5, 128, dropout, (0,0), kernel_init)],
                    [(5, 64, 0, (0,0), kernel_init), (5, 64, dropout, reg, kernel_init), (1, 1, dropout, reg, kernel_init)],
                ],
            },
            'small_net': {
                'down_conv_kernels': [
                    [(5, 64, 0, (0,0), kernel_init), (5, 64, dropout, (0,0), kernel_init)],
                    [(5, 128, 0, (0,0), kernel_init), (5, 128, dropout, reg, kernel_init)],
                    [(5, 256, 0, (0,0), kernel_init), (5, 256, dropout, reg, kernel_init)],
                ],
                'lstm_kernels': [
                    [(5, 64, 0, (0,0), kernel_init)],
                    [(5, 128, 0, (0,0), kernel_init)],
                    [(5, 256, dropout, reg, kernel_init)],
                ],
                'up_conv_kernels': [
                    [(5, 256, 0, (0,0), kernel_init), (5, 256, dropout, (0,0), kernel_init)],
                    [(5, 128, 0, (0,0), kernel_init), (5, 128, dropout, (0,0), kernel_init)],
                    [(5, 64, 0, (0,0), kernel_init), (5, 64, dropout, reg, kernel_init), (1, 1, dropout, reg, kernel_init)],
                ],
            },
            'shorter_net': {
                'down_conv_kernels': [
                    [(5, 128, 0, (0,0), kernel_init), (5, 128, dropout, reg, kernel_init)],
                    [(5, 256, 0, (0,0), kernel_init), (5, 256, dropout, (0,0), kernel_init)],
                    [(5, 512, 0, (0,0), kernel_init), (5, 512, dropout, reg, kernel_init)],
                ],
                'lstm_kernels': [
                    [(5, 128, dropout, (0,0), kernel_init)],
                    [(5, 256, dropout, reg, kernel_init)],
                    [(5, 512, dropout, (0,0), kernel_init)],
                ],
                'up_conv_kernels': [
                    [(5, 256, 0, (0,0), kernel_init), (5, 256, dropout, reg, kernel_init)],
                    [(5, 128, 0, (0,0), kernel_init), (5, 128, dropout, (0,0), kernel_init)],
                    [(5, 64, 0, (0,0), kernel_init), (5, 64, dropout, reg, kernel_init), (1, 1, dropout, reg, kernel_init)],
                ],
            },
            'longLSTM_net': {
                'down_conv_kernels': [
                    [(5, 128, 0, reg, kernel_init), (5, 128, dropout, reg, kernel_init)],
                    [(5, 256, 0, reg, kernel_init), (5, 256, dropout, reg, kernel_init)],
                    [(5, 512, 0, reg, kernel_init), (5, 512, dropout, reg, kernel_init)],
                ],
                'lstm_kernels': [
                    [(5, 64, dropout, reg, kernel_init), (5, 64, dropout, reg, kernel_init)],
                    [(5, 128, dropout, reg, kernel_init), (5, 128, dropout, reg, kernel_init)],
                    [(5, 256, dropout, reg, kernel_init), (5, 256, dropout, reg, kernel_init)],
                ],
                'up_conv_kernels': [
                    [(5, 256, 0, reg, kernel_init), (5, 256, dropout, reg, kernel_init)],
                    [(5, 128, 0, reg, kernel_init), (5, 128, dropout, reg, kernel_init)],
                    [(5, 64, 0, reg, kernel_init), (5, 64, dropout, reg, kernel_init), (1, 1, dropout, reg, kernel_init)],
                ],
            }
    }
    return net_kernel_params 


def TrainableLayers(model, option):
    if option == 'none':
        model.trainable = False
    
    if option == 'full':
        model.trainable = True
        
    if option == 'only_enc':
        for i, block in enumerate(model.layers):
            if i < 4:
                block.trainable = True
            else:
                block.Trainable = False
    
    if option == 'only_dec':
        for i, block in enumerate(model.layers):
            if i > 3:
                block.trainable = True
            else:
                block.Trainable = False
    return model
               