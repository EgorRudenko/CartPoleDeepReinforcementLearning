extends Sprite2D


var rng = RandomNumberGenerator.new() # random number generator
var radius = 1;
var angularSpeed = 0;
var airFrictionCoefficient = 1;
var speedX =0;					# actual speed in direction X
var mass = 50;					# you can guess
var externalForce;				# something like little wind. Without it block wouldn't fall if not touched (it isn't used now?)
var forceX = 0; 				# Force in X direction, including projection of gravity
var gameOver = 0;				# I hope you can guess
var action = 0;					# variable to save an action chosen by AI
var timeMultiplier = 1;


var client = WebSocketPeer.new();
var url = "ws://localhost:5000";


func projectionOntoTangent(theta:float, a:float) -> float:
	return sin(theta)*a
func projectionOntoX(theta:float, a:float) -> float:
	return sin(theta)*cos(theta)*a
func movement(delta:float, action:int) -> void:
	var theta = rotation
	var gravity = 10 * mass
	var tanForce = projectionOntoTangent(theta, gravity) 
	tanForce -= airFrictionCoefficient*(angularSpeed**2)*sign(angularSpeed)
	forceX = action*1000 - projectionOntoX(theta, gravity) - airFrictionCoefficient*(speedX**2)*sign(speedX)
	tanForce -= forceX * cos(theta)
	speedX += forceX*delta/mass
	position += Vector2(speedX, 0)
	var angularAcceleration = tanForce / (radius*mass)
	angularSpeed += angularAcceleration * delta
	rotation += angularSpeed * delta
	
	if rotation > PI/2 or rotation < -PI/2:
		rotation = PI/2 if rotation + angularSpeed > PI/2 else -PI/2
		speedX/=1000
		gameOver = 1
	if (abs(position[0]) >= 1000):
		gameOver = 1
func determineActionFromKeyboard() -> int:
	var a = 0
	if Input.is_action_pressed("left"):
		a += -1
	if Input.is_action_pressed("right"):
		a += 1
	return a
func communicate() -> void:
	client.poll()
	var state = client.get_ready_state()
	if state == WebSocketPeer.STATE_OPEN:
		client.send_text(str(rotation/(PI/2), " ", speedX/5, " ", angularSpeed/0.04, " ", gameOver))
		action = 0;
		while client.get_available_packet_count():
			action = int(client.get_packet().get_string_from_utf8())
	elif state == WebSocketPeer.STATE_CLOSED:
		#var code = client.get_close_code()
		#var reason = client.get_close_reason()
		#print("WebSocket closed with code: %d, reason %s. Clean: %s" % [code, reason, code != -1])
		set_process(false) # Stop processing.
func reinit() -> void:							# return to the beginning of the game
	angularSpeed = 0;
	speedX =0
	position = Vector2(0,0)
	rotation = 0
	forceX = 0
	gameOver = 0

func _ready():
	#client.connect("data_received", f)
	var err = client.connect_to_url(url);
	

func _process(delta):
	movement(delta*timeMultiplier, action)
	communicate()
	if (gameOver == 1):
		reinit()
		
