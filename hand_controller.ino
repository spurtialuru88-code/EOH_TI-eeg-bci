/*
 * hand_controller.ino — Arduino Firmware for Bionic Hand Control
 * 
 * UIUC EOH — NeuroGrip BCI Bionic Hand
 * 
 * Receives single-character commands from Python over serial:
 *   'O' → Open hand (extend all fingers)
 *   'C' → Close hand (flex all fingers — grasp)
 *   'P' → Pinch grip (thumb + index only)
 *   'N' → Neutral position
 *   'V' → Proportional mode (next bytes = 0-100 value)
 * 
 * WIRING:
 *   Servo 1 (thumb)  → Pin 3
 *   Servo 2 (index)  → Pin 5
 *   Servo 3 (middle) → Pin 6
 *   Servo 4 (ring)   → Pin 9
 *   Servo 5 (pinky)  → Pin 10
 *   
 *   If using a single servo for all fingers (tendon-driven):
 *     Main servo → Pin 9
 *   
 *   Power: Servos need 5-6V from external supply, NOT from Arduino 5V pin!
 *          Connect servo power to external supply, GND to Arduino GND.
 * 
 * SERVO ANGLES:
 *   0°   = finger fully extended (open)
 *   180° = finger fully flexed (closed)
 *   Adjust OPEN_ANGLE and CLOSE_ANGLE below for your hand design.
 */

#include <Servo.h>

// ============================================================
// CONFIGURATION — Adjust these for YOUR hand
// ============================================================

// Single servo mode (one servo controls all fingers via tendons)
#define SINGLE_SERVO_MODE true

// Pin assignments
#define SERVO_MAIN_PIN   9    // Single servo mode
#define SERVO_THUMB_PIN  3    // Multi-servo mode
#define SERVO_INDEX_PIN  5
#define SERVO_MIDDLE_PIN 6
#define SERVO_RING_PIN   9
#define SERVO_PINKY_PIN  10

// Angle limits (tune these for your specific hand)
#define OPEN_ANGLE    10     // Fully open
#define CLOSE_ANGLE   170    // Fully closed
#define NEUTRAL_ANGLE 90     // Rest position
#define PINCH_ANGLE   130    // Partial close for pinch

// Movement speed (lower = slower, smoother)
#define MOVE_DELAY_MS 15     // ms between each degree of movement

// Serial
#define BAUD_RATE 9600

// ============================================================
// GLOBALS
// ============================================================

#if SINGLE_SERVO_MODE
  Servo mainServo;
#else
  Servo thumbServo;
  Servo indexServo;
  Servo middleServo;
  Servo ringServo;
  Servo pinkyServo;
#endif

int currentAngle = NEUTRAL_ANGLE;
int targetAngle = NEUTRAL_ANGLE;
char currentState = 'N';

// Status LED
#define LED_PIN 13

// ============================================================
// SETUP
// ============================================================

void setup() {
  Serial.begin(BAUD_RATE);
  
  pinMode(LED_PIN, OUTPUT);
  
  #if SINGLE_SERVO_MODE
    mainServo.attach(SERVO_MAIN_PIN);
    mainServo.write(NEUTRAL_ANGLE);
  #else
    thumbServo.attach(SERVO_THUMB_PIN);
    indexServo.attach(SERVO_INDEX_PIN);
    middleServo.attach(SERVO_MIDDLE_PIN);
    ringServo.attach(SERVO_RING_PIN);
    pinkyServo.attach(SERVO_PINKY_PIN);
    
    thumbServo.write(NEUTRAL_ANGLE);
    indexServo.write(NEUTRAL_ANGLE);
    middleServo.write(NEUTRAL_ANGLE);
    ringServo.write(NEUTRAL_ANGLE);
    pinkyServo.write(NEUTRAL_ANGLE);
  #endif
  
  currentAngle = NEUTRAL_ANGLE;
  
  // Startup sequence — visual confirmation
  blinkLED(3, 200);
  
  Serial.println("HAND_READY");
}

// ============================================================
// MAIN LOOP
// ============================================================

void loop() {
  // Check for commands from Python
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    processCommand(cmd);
  }
  
  // Smooth movement toward target angle
  smoothMove();
  
  delay(1);
}

// ============================================================
// COMMAND PROCESSING
// ============================================================

void processCommand(char cmd) {
  switch (cmd) {
    case 'O':  // Open
      targetAngle = OPEN_ANGLE;
      currentState = 'O';
      digitalWrite(LED_PIN, LOW);
      Serial.println("ACK:OPEN");
      break;
      
    case 'C':  // Close
      targetAngle = CLOSE_ANGLE;
      currentState = 'C';
      digitalWrite(LED_PIN, HIGH);
      Serial.println("ACK:CLOSE");
      break;
      
    case 'P':  // Pinch
      targetAngle = PINCH_ANGLE;
      currentState = 'P';
      digitalWrite(LED_PIN, HIGH);
      Serial.println("ACK:PINCH");
      
      #if !SINGLE_SERVO_MODE
        // Pinch: only thumb + index close, others stay open
        smoothMoveTo(thumbServo, CLOSE_ANGLE);
        smoothMoveTo(indexServo, CLOSE_ANGLE);
        smoothMoveTo(middleServo, OPEN_ANGLE);
        smoothMoveTo(ringServo, OPEN_ANGLE);
        smoothMoveTo(pinkyServo, OPEN_ANGLE);
      #endif
      break;
      
    case 'N':  // Neutral
      targetAngle = NEUTRAL_ANGLE;
      currentState = 'N';
      digitalWrite(LED_PIN, LOW);
      Serial.println("ACK:NEUTRAL");
      break;
      
    case 'V':  // Proportional (0-100)
      {
        // Read the value string (e.g., "V75\n")
        String valStr = Serial.readStringUntil('\n');
        int val = valStr.toInt();
        val = constrain(val, 0, 100);
        
        // Map 0-100 → OPEN_ANGLE to CLOSE_ANGLE
        targetAngle = map(val, 0, 100, OPEN_ANGLE, CLOSE_ANGLE);
        currentState = 'V';
        
        Serial.print("ACK:PROP:");
        Serial.println(val);
      }
      break;
      
    default:
      // Ignore unknown commands
      break;
  }
}

// ============================================================
// SMOOTH MOVEMENT
// ============================================================

void smoothMove() {
  /*
   * Move one degree toward the target each cycle.
   * This prevents jerky movement and reduces servo stress.
   * 
   * At MOVE_DELAY_MS = 15, full range (0-180°) takes ~2.7 seconds.
   * For faster response, decrease MOVE_DELAY_MS.
   */
  
  if (currentAngle == targetAngle) return;
  
  if (currentAngle < targetAngle) {
    currentAngle++;
  } else {
    currentAngle--;
  }
  
  #if SINGLE_SERVO_MODE
    mainServo.write(currentAngle);
  #else
    // In multi-servo mode, move all fingers together
    // (except during pinch, handled separately)
    if (currentState != 'P') {
      thumbServo.write(currentAngle);
      indexServo.write(currentAngle);
      middleServo.write(currentAngle);
      ringServo.write(currentAngle);
      pinkyServo.write(currentAngle);
    }
  #endif
  
  delay(MOVE_DELAY_MS);
}

#if !SINGLE_SERVO_MODE
void smoothMoveTo(Servo &servo, int target) {
  int current = servo.read();
  while (current != target) {
    if (current < target) current++;
    else current--;
    servo.write(current);
    delay(MOVE_DELAY_MS);
  }
}
#endif

// ============================================================
// UTILITIES
// ============================================================

void blinkLED(int times, int delayMs) {
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(delayMs);
    digitalWrite(LED_PIN, LOW);
    delay(delayMs);
  }
}
