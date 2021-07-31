# Stock-Market-Prediction-Using-LSTM-Deep-Learning
A deep learning algorithm used to make stock market predictions on historic data analysis.

## Introduction
Artificial Intelligence (A.I) and Machine Learning (ML) have been transforming finance and the investing world. They made automated trading possible, and allowed many hedgefunds and banks to maximize profit margins with relative low risk through their algorithmic trading and time series analysis capabilities.

A huge benefit to using machine learning in finance and investing is that these AI powered robo-advisers can perform real-time analysis on massive datasets and trade securities at an extremely faster rate compared to human traders.


## Implementation
The program uses Recurrent-Neural-Networks, rather than the traditional Deep Neural Networks, or more commonly known as "Vanilla Networks" by enthursiasts. This is because using the regular Feed-Forward Neural Network would map a fixed size input to a fixed size output. These neural networks don't have any time-dependency or memory affect in their data, **however** RNNs do, and therefore when it comes to time dependency data, the use of RNNs would be more preferred over regular ANNs. Although RNNs do have a slight problem....

### Vanishing Gradient Problem
This is essentially an error which occures during the training of RNNs, when over multiple epochs of backpropogation, these deep learning methods use what is known as the **gradient descent method** to better fine-tune their weights. Now the more layers that are added, the more the gradients of the loss function approahc zero, making the network hard to train.
<br><br>
This mainly occurs during **backpropogation**, where we calculate the derivaties of the network by moving from the outermost layer, back to the initial later. The chain rule is used during this calculation in which the derivates from the final layers are multiplied by the derivatives from early layers. **The gradient keeps diminishing exponentially and therefore the weights and biases are no longer being updated.**

### Long-Short-Term-Memory Neural Networks
To avoid the **Vanishing Gradient Problem** the LSTM network was implemented, LSTM networks are type of RNN that are designed to remember long term dependencies by default. LSTM can remember and recall information for a prolonged period of time, which is why they were used in the program as they would recall information and are a good choice due to their recurrnt nature to be used in time-dependency data such as making stock-market predictions


## Libraries 
To be able to use this algorithm, the following libraries must be present:
<br>
• **pandas_datareader<br>**
• **pandas<br>**
• **os<br>**
• **plotly<br>**
• **sklearn<br>**
• **numpy<br>**
• **tensorflow**<br>

this can be done by opening cmd > then typing **pip install** followed by the library name if those libraries aren't present

 
## Datasources
The algorithm relied on using pandas_datareader.data(), and the **yahoo** opensource API to get the historic stockmarket date as the default source, however many other sources can be used in place of yahoo, such as:
<br>
• **Tiingo<br> 
• IEX<br>
• Alpha Vantage<br>
• Econdb<br>
• World Bank<br>
• OECD<br>
• Eurostat<br>
• Nasdaq Trader symbol definitions<br>
• Yahoo Finance<br>**
for more options on using the pandas_datareader, use this website: 
https://pandas-datareader.readthedocs.io/en/latest/remote_data.html#remote-data-econdb



## Future & Contributions
Currently working on better optimizing the program, and planning to add features such as:
• **Perform potential forecasting on stock prices**
• **Implement more deep learning methods**
• **Increase performance of program**
• **Implement it using Java**


Note: This is opensource, so you are more than happy to use my code to help you in finance and stock related projects, also any contributions would be greatly appreciated :)


