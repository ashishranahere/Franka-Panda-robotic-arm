# Simple PID controller (no class)

# Initialize variables
kp = 1.0
ki = 0.1
kd = 0.05
dt = 0.1

integral = 0
prev_error = 0
prev_derivative = 0

alpha = 0.2   # for derivative smoothing


def pid_compute(setpoint, measurement):
    global integral, prev_error, prev_derivative

    error = setpoint - measurement

    # Proportional
    P = kp * error

    # Integral (with anti-windup)
    integral += error * dt
    if integral > 10:
        integral = 10
    elif integral < -10:
        integral = -10
    I = ki * integral

    # Derivative (with filtering)
    raw_derivative = (error - prev_error) / dt
    derivative = alpha * raw_derivative + (1 - alpha) * prev_derivative
    D = kd * derivative

    # Save for next iteration
    prev_error = error
    prev_derivative = derivative

    return P + I + D


# Simple test
if __name__ == "__main__":
    position = 0
    target = 10

    for i in range(50):
        control = pid_compute(target, position)

        # simulate system response
        position += control * 0.1

        print(f"Step {i}: Position = {position:.2f}, Control = {control:.2f}")