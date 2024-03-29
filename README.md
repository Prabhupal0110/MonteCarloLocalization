# Monte Carlo Localization
Monte Carlo Algorithm(Xt-1, ut, zt):\
&nbsp;&nbsp;&nbsp;	X_t = X ̅_t = Ф\
&nbsp;&nbsp;&nbsp;	For n=1 to N:\
&nbsp;&nbsp;&nbsp;	&nbsp;&nbsp;&nbsp;	x_t^n = motion_update(u_t, x_(t-1)^n)\
&nbsp;&nbsp;&nbsp;	&nbsp;&nbsp;&nbsp;	w_t^n = sensor_update(z_t  , x_t^n)\
&nbsp;&nbsp;&nbsp;	&nbsp;&nbsp;&nbsp;	X ̅_t = X ̅_t + (x_t^n,w_t^n)\
&nbsp;&nbsp;&nbsp;            End\
&nbsp;&nbsp;&nbsp;	#Re- Sampling of the particles on the basis of the weights\
 &nbsp;&nbsp;&nbsp;           For n=1 to N:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;		Draw x_t^n from X ̅_t with probability ∝ w_t^n\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;		X_t= X_t + x_t^n\
&nbsp;&nbsp;&nbsp;	End\
&nbsp;&nbsp;&nbsp;	Return X_t\

Motion Update:
As the robot moves for each time iteration, its position and orientation keeps on changing, therefore it is necessary that the particles also move the same distance and direction. Also we know the map of the environment, so we can use the odometry to find out the new position and orientation of particles in the map. It is to be noted that there will be some noise when it comes to odometry estimation. 

Sensor Update:
Once the particles are moved (using odometry) as per the distance and turning angle of the actual robot, now for each particle, the probability is calculated of how accurate the robot would have sensed the obstacles given it is at the state of that particle, Now, this probability is assigned to each particle as weight w_t^n.

Re Sampling:
Now the N particles are again drawn randomly from the previous belief but with the probability proportional to w_t^n. As a result, the particles that are consistent with the readings of sensor are likely to be drawn more. 
NOTE: This part Re Sampling is yet to be implemented in the code, however the result was verified by plotting particle with highest weight and found quite close to the actual robot position. (Results included in next part)


# RESULTS:
Iteration:  1, Errors:  25.603993280308973\
Iteration:  2, Errors:  13.065813822693489\
Iteration:  3, Errors:  8.500059694407774\
Iteration:  4, Errors:  9.314921704622229\
Iteration:  5, Errors:  5.561024715550335\
Iteration:  6, Errors:  4.253458916589124\
Iteration:  7, Errors:  5.981728526489809\
Iteration:  8, Errors:  5.758568050234015\
Iteration:  9, Errors:  4.399428595062176\
Iteration:  10, Errors:  6.888019455303352\
Iteration:  11, Errors:  6.034441108578674\
Iteration:  12, Errors:  4.899607317144078\
Iteration:  13, Errors:  5.9036627180645755\
Iteration:  14, Errors:  3.82114936628902\
Iteration:  15, Errors:  5.84355294030965\
Iteration:  16, Errors:  4.740471615850745\
Iteration:  17, Errors:  5.735789231521912\
Iteration:  18, Errors:  2.0434386235406348\
Iteration:  19, Errors:  5.331134662299528\
Iteration:  20, Errors:  1.969549367967016\

![alt text](https://github.com/Prabhupal0110/MonteCarloLocalization/blob/main/Capture.PNG) \\
![alt text](https://github.com/Prabhupal0110/MonteCarloLocalization/blob/main/Figure%202022-05-02%20123559%20(21).png)
