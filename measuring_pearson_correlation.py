import numpy as np

# Data for MNIST
stragglers_MNIST = [56.40, 63.43, 192.93, 200.23, 114.43, 202.73, 77.57, 130.5, 194.7, 192.2]
confidence_MNIST = [72.57, 80.4, 181, 171.63, 104, 206.73, 92.37, 120.7, 239.23, 156.5]
energy_MNIST = [77.2, 81.87, 178.07, 162.5, 105.67, 203, 90.97, 115.13, 256.4, 154.33]
class_level_error_MNIST = [1 - 0.96237177, 1 - 0.9723452, 1 - 0.90329546, 1 - 0.89264405, 1 - 0.92276675,
                           1 - 0.8669742, 1 - 0.9461784, 1 - 0.92778397, 1 - 0.8786414, 1 - 0.90164125]
# Data for KMNIST
stragglers_KMNIST = [78.3, 406.63, 615.43, 253.5, 400.57, 354.67, 358.2, 251.33, 299.33, 302.47]
class_level_error_KMNIST = [1 - 0.88416785, 1 - 0.7879815, 1 - 0.7151908, 1 - 0.8570282, 1 - 0.77458483, 1 - 0.8273136,
                            1 - 0.798144, 1 - 0.86421686, 1 - 0.83318824, 1 - 0.83617574]
# Data for FashionMNIST
stragglers_FashionMNIST = [325.57, 106.93, 611.7, 247.97, 401.5, 216.93, 1195.17, 217.37, 127.87, 132.7]
class_level_error_FashionMNIST = [1 - 0.83218604, 1 - 0.9593086, 1 - 0.7372085, 1 - 0.8558186, 1 - 0.77769,
                                  1 - 0.90594727, 1 - 0.53281194, 1 - 0.9025393, 1 - 0.9416273, 1 - 0.93206877]

# Calculate Pearson correlation coefficient
correlation1 = np.corrcoef(stragglers_MNIST, class_level_error_MNIST)[0, 1]
correlation2 = np.corrcoef(confidence_MNIST, class_level_error_MNIST)[0, 1]
correlation3 = np.corrcoef(energy_MNIST, class_level_error_MNIST)[0, 1]
correlation4 = np.corrcoef(stragglers_KMNIST, class_level_error_KMNIST)[0, 1]
correlation5 = np.corrcoef(stragglers_FashionMNIST, class_level_error_FashionMNIST)[0, 1]

print(f'Pearson correlation coefficients on MNIST:\n'
      f'  - stragglers vs class-level error: {correlation1},\n'
      f'  - confidence vs class-level error: {correlation2},\n'
      f'  - energy vs class-level error: {correlation3}\n'
      f'Pearson correlation coefficients on KMNIST:\n'
      f'  - stragglers vs class-level error: {correlation4}\n'
      f'Pearson correlation coefficients on FashionMNIST:\n'
      f'  - stragglers vs class-level error: {correlation5}')
