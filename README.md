# MonteCarloLocalization
Monte Carlo Algorithm(Xt-1, ut, zt):
	X_t = X ̅_t = Ф
	For n=1 to N:
		x_t^n = motion_update(u_t, x_(t-1)^n)
		w_t^n = sensor_update(z_t  , x_t^n)
		X ̅_t = X ̅_t + (x_t^n,w_t^n)
            End
	#Re- Sampling of the particles on the basis of the weights
            For n=1 to N:
		Draw x_t^n from X ̅_t with probability ∝ w_t^n
		X_t= X_t + x_t^n
	End
	Return X_t

Motion Update:
As the robot moves for each time iteration, its position and orientation keeps on changing, therefore it is necessary that the particles also move the same distance and direction. Also we know the map of the environment, so we can use the odometry to find out the new position and orientation of particles in the map. It is to be noted that there will be some noise when it comes to odometry estimation. 

Sensor Update:
Once the particles are moved (using odometry) as per the distance and turning angle of the actual robot, now for each particle, the probability is calculated of how accurate the robot would have sensed the obstacles given it is at the state of that particle, Now, this probability is assigned to each particle as weight w_t^n.
Re Sampling:
Now the N particles are again drawn randomly from the previous belief but with the probability proportional to w_t^n. As a result, the particles that are consistent with the readings of sensor are likely to be drawn more. 
NOTE: This part Re Sampling is yet to be implemented in the code, however the result was verified by plotting particle with highest weight and found quite close to the actual robot position. (Results included in next part)
