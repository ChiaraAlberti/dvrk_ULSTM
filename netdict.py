"""
Created on Thu Dec 12 15:45:07 2019

@author: stormlab
"""
def Net_type(self, dropout, reg, kernel_init):
    net_kernel_params = {
            'cpu_net': {            
                'down_conv_kernels': [
                    [(5, 128, dropout, reg, kernel_init), (5, 128, dropout, reg, kernel_init)],  
                    [(5, 256, dropout, reg, kernel_init), (5, 256, dropout, reg, kernel_init)],
                    [(5, 256, dropout, reg, kernel_init), (5, 256, dropout, reg, kernel_init)],
                    [(5, 512, dropout, reg, kernel_init), (5, 512, dropout, reg, kernel_init)],
                ],
                'lstm_kernels': [
                    [(5, 128, dropout, reg, kernel_init)], 
                    [(5, 256, dropout, reg, kernel_init)],
                    [(5, 256, dropout, reg, kernel_init)],
                    [(5, 512, dropout, reg, kernel_init)],
                ],
                'up_conv_kernels': [
                    [(5, 256, dropout, reg, kernel_init), (5, 256, dropout, reg, kernel_init)],
                    [(5, 128, dropout, reg, kernel_init), (5, 128, dropout, reg, kernel_init)],
                    [(5, 64, dropout, reg, kernel_init), (5, 64, dropout, reg, kernel_init)],
                    [(5, 32, dropout, reg, kernel_init), (5, 32, dropout, reg, kernel_init), (1, 1, dropout, reg, kernel_init)],
                ],
            },   
            'deeper net': {
                'down_conv_kernels': [
                    [(5, 50, dropout, reg, kernel_init), (5, 50, dropout, reg, kernel_init)],
                    [(5, 100, dropout, reg, kernel_init), (5, 100, dropout, reg, kernel_init)],
                    [(5, 100, dropout, reg, kernel_init), (5, 100, dropout, reg, kernel_init)],
                    [(5, 200, dropout, reg, kernel_init), (5, 200, dropout, reg, kernel_init)],
                    [(5, 400, dropout, reg, kernel_init), (5, 400, dropout, reg, kernel_init)],
                ],
                'lstm_kernels': [
                    [(5, 50, dropout, reg, kernel_init)],
                    [(5, 100, dropout, reg, kernel_init)],
                    [(5, 100, dropout, reg, kernel_init)],
                    [(5, 200, dropout, reg, kernel_init)],
                    [(5, 400, dropout, reg, kernel_init)],                
                ],
                'up_conv_kernels': [
                    [(5, 100, dropout, reg, kernel_init), (5, 100, dropout, reg, kernel_init)],
                    [(5, 100, dropout, reg, kernel_init), (5, 100, dropout, reg, kernel_init)],
                    [(5, 50, dropout, reg, kernel_init), (5, 50, dropout, reg, kernel_init)],
                    [(5, 50, dropout, reg, kernel_init), (5, 50, dropout, reg, kernel_init)],
                    [(5, 20, dropout, reg, kernel_init), (5, 20, dropout, reg, kernel_init), (1, 1, dropout, reg, kernel_init)],
                ],
            },
            'original net': {
                'down_conv_kernels': [
                    [(5, 100, dropout, reg, kernel_init), (5, 100, dropout, reg, kernel_init)],
                    [(5, 200, dropout, reg, kernel_init), (5, 200, dropout, reg, kernel_init)],
                    [(5, 200, dropout, reg, kernel_init), (5, 200, dropout, reg, kernel_init)],
                    [(5, 400, dropout, reg, kernel_init), (5, 400, dropout, reg, kernel_init)],
                ],
                'lstm_kernels': [
                    [(5, 100, dropout, reg, kernel_init)],
                    [(5, 200, dropout, reg, kernel_init)],
                    [(5, 200, dropout, reg, kernel_init)],
                    [(5, 400, dropout, reg, kernel_init)],
                ],
                'up_conv_kernels': [
                    [(5, 200, dropout, reg, kernel_init), (5, 200, dropout, reg, kernel_init)],
                    [(5, 100, dropout, reg, kernel_init), (5, 100, dropout, reg, kernel_init)],
                    [(5, 50, dropout, reg, kernel_init), (5, 50, dropout, reg, kernel_init)],
                    [(5, 20, dropout, reg, kernel_init), (5, 20, dropout, reg, kernel_init), (1, 1, dropout, reg, kernel_init)],
                ],
            },
            'shorter net': {
                'down_conv_kernels': [
                    [(5, 128, dropout, reg, kernel_init), (5, 128, dropout, reg, kernel_init)],
                    [(5, 256, dropout, reg, kernel_init), (5, 256, dropout, reg, kernel_init)],
                    [(5, 512, dropout, reg, kernel_init), (5, 512, dropout, reg, kernel_init)],
                ],
                'lstm_kernels': [
                    [(5, 128, dropout, reg, kernel_init)],
                    [(5, 256, dropout, reg, kernel_init)],
                    [(5, 512, dropout, reg, kernel_init)],
                ],
                'up_conv_kernels': [
                    [(5, 256, dropout, reg, kernel_init), (5, 256, dropout, reg, kernel_init)],
                    [(5, 128, dropout, reg, kernel_init), (5, 128, dropout, reg, kernel_init)],
                    [(5, 64, dropout, reg, kernel_init), (5, 64, dropout, reg, kernel_init), (1, 1, dropout, reg, kernel_init)],
                ],
            },
            'longLSTM net': {
                'down_conv_kernels': [
                    [(5, 128, dropout, reg, kernel_init), (5, 128, dropout, reg, kernel_init)],
                    [(5, 256, dropout, reg, kernel_init), (5, 256, dropout, reg, kernel_init)],
                    [(5, 512, dropout, reg, kernel_init), (5, 512, dropout, reg, kernel_init)],
                ],
                'lstm_kernels': [
                    [(5, 64, dropout, reg, kernel_init)],
                    [(5, 128, dropout, reg, kernel_init)],
                    [(5, 128, dropout, reg, kernel_init)],
                    [(5, 256, dropout, reg, kernel_init)],
                    [(5, 512, dropout, reg, kernel_init)],
                ],
                'up_conv_kernels': [
                    [(5, 256, dropout, reg, kernel_init), (5, 256, dropout, reg, kernel_init)],
                    [(5, 128, dropout, reg, kernel_init), (5, 128, dropout, reg, kernel_init)],
                    [(5, 64, dropout, reg, kernel_init), (5, 64, dropout, reg, kernel_init), (1, 1, dropout, reg, kernel_init)],
                ],
            }
    }