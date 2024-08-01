extends Sprite2D


var rng = RandomNumberGenerator.new() # random number generator
var speedR = 0;					# angular speed of the bloke
var speedX =0;					# actual speed in direction X
var mass = 50;					# you can guess
var forceY = 0;					# force tangent to rotation including gravity and projection of forceX
var externalForce;				# something like little wind. Without it block wouldn't fall
var frictionCoefficient = 400;	# isn't scientific i just substract dt*v*this_coeff to add some energy loss
var forceX = 0; 				# Force in X direction, including projection of gravity
var gameOver = 0;				# I hope you can guess


var action = 0;


@warning_ignore("shadowed_variable")
func gravity(mass, theta):
	return [10*mass*cos(theta-PI/2),10*mass*(sin(theta-PI/2)*cos(theta-PI/2))]
func movement(delta, action):
	#externalForce = rng.randf_range(-10, 10);
	externalForce = 0
	
	forceY = externalForce + gravity(mass, rotation)[0];
	#if action == 1: # left
	#	forceX -= 100000*delta;
	#if action == -1: # right
	#	forceX += 100000*delta;


	#forceX -= 100 * delta * ( -1 if forceX < 0 else 1)

	
	speedX += (gravity(mass, rotation)[1]+action*1000)*delta/mass
	
	position += Vector2(speedX, 0)
	
	speedR += forceY*delta/mass + sin(rotation-PI/2)*speedX/PI/20
	#print(speedR)
	speedR -= frictionCoefficient*delta*speedR
	#print(speedR)
	#var velocity = Vector2.UP.rotated(rotation) * speed
	if abs(speedX) > 5:
		speedX = 5 * (speedX/abs(speedX))
	#if abs(speedR) > 0.04:
	#	speedR = 0.04 * (speedR/abs(speedR))
	
	
	if (not (rotation + speedR > PI/2) and not (rotation + speedR < -PI/2)):
		rotation += speedR
	else:
		rotation = PI/2 if rotation + speedR > PI/2 else -PI/2
		speedX/=1000
		gameOver = 1
	if (abs(position[0]) >= 1000):
		gameOver = 1
func determineActionFromKeyboard():
	var a = 0
	if Input.is_action_pressed("left"):
		a += -1
	if Input.is_action_pressed("right"):
		a += 1
	return a
func reinit():							# return to the beginning of the game
	speedR = 0
	speedX =0
	position = Vector2(0,0)
	rotation = 0
	forceY = 0
	frictionCoefficient = 100
	forceX = 0
	gameOver = 0


# a little about idea
# player may not fall and should be as close as possible to center
# I will try to make AI play this


var client = WebSocketPeer.new();
var url = "ws://localhost:5000";


func communicate():
	client.poll()
	var state = client.get_ready_state()
	if state == WebSocketPeer.STATE_OPEN:
		client.send_text(str(rotation/(PI/2), " ", speedX/5, " ", speedR/0.04, " ", gameOver))
		while client.get_available_packet_count():
			action = int(client.get_packet().get_string_from_utf8())
	elif state == WebSocketPeer.STATE_CLOSED:
		#var code = client.get_close_code()
		#var reason = client.get_close_reason()
		#print("WebSocket closed with code: %d, reason %s. Clean: %s" % [code, reason, code != -1])
		set_process(false) # Stop processing.


func _ready():
	#client.connect("data_received", f)
	var err = client.connect_to_url(url);
	print(err)
	

func _process(delta):
	movement(delta, action)
	communicate()
	if (gameOver == 1):
		reinit()
		
