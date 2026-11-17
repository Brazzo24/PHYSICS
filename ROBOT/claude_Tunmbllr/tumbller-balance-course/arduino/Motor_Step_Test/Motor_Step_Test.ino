/*
 * Motor_Step_Test - open-loop motor / encoder / battery test for the ELEGOO Tumbller
 * (Tumbller balance course, notebook 4)
 *
 *   !!  PUT THE ROBOT ON A STAND SO THAT BOTH WHEELS SPIN FREELY.  !!
 *   !!  This sketch does NOT balance. The wheels run at full speed. !!
 *
 * Serial Monitor: 115200 baud. Type one letter and press Enter:
 *   c   print the encoder edge counter of the left wheel
 *       (turn the wheel by hand exactly 10 turns -> edges per wheel revolution)
 *   z   set the counter to zero
 *   v   print the battery voltage
 *   g   run the step sequence. Prints one line every 10 ms:   ms,pwm,count
 *       Copy the lines into a file, or record them with tools/tumbller_logger.py
 *
 * The counter counts both edges of ONE encoder channel, exactly like the stock
 * firmware does (attachInterrupt(..., CHANGE)). It cannot see the direction.
 */

// Pin numbers copied from Pins.h of the stock firmware
#define AIN1 7                 // direction, left motor
#define PWMA_LEFT 5            // PWM, left motor
#define BIN1 12                // direction, right motor
#define PWMB_RIGHT 6           // PWM, right motor
#define STBY_PIN 8             // HIGH = motor driver enabled
#define ENCODER_LEFT_A_PIN 2   // left encoder, channel A (external interrupt 0)
#define VOL_MEASURE_PIN A2     // battery voltage through a 10k / 1.5k divider

const int STEP_PWM[] = {60, 120, 180, 240};   // PWM values of the test steps
const int NUM_STEPS = 4;
const unsigned long STEP_ON_MS = 700;         // motor on
const unsigned long STEP_OFF_MS = 500;        // motor off (coast down)
const unsigned long SAMPLE_MS = 10;           // one line every 10 ms

volatile long edge_count = 0;   // written inside the interrupt -> "volatile"

void countEdge()                // interrupt service routine: keep it tiny
{
  edge_count++;
}

long readCount()
{
  noInterrupts();               // a long is 4 bytes; do not let the ISR change it half-way
  long c = edge_count;
  interrupts();
  return c;
}

void setMotors(int pwm)         // pwm: -255 ... 255, sign = direction
{
  digitalWrite(AIN1, pwm < 0 ? HIGH : LOW);
  digitalWrite(BIN1, pwm < 0 ? HIGH : LOW);
  analogWrite(PWMA_LEFT, abs(pwm));
  analogWrite(PWMB_RIGHT, abs(pwm));
}

float batteryVoltage()          // same formula as voltage.h of the stock firmware
{
  return (analogRead(VOL_MEASURE_PIN) * 1.1 / 1024) * ((10 + 1.5) / 1.5);
}

void logFor(unsigned long duration_ms, int pwm, unsigned long t0)
{
  setMotors(pwm);
  unsigned long start = millis();
  unsigned long next = start;
  while (millis() - start < duration_ms)
  {
    if (millis() >= next)
    {
      next += SAMPLE_MS;
      Serial.print(millis() - t0);
      Serial.print(',');
      Serial.print(pwm);
      Serial.print(',');
      Serial.println(readCount());
    }
  }
}

void runStepSequence()
{
  Serial.print(F("# battery [V]: "));
  Serial.println(batteryVoltage());
  Serial.println(F("ms,pwm,count"));
  noInterrupts();
  edge_count = 0;
  interrupts();
  unsigned long t0 = millis();
  logFor(200, 0, t0);
  for (int i = 0; i < NUM_STEPS; i++)
  {
    logFor(STEP_ON_MS, STEP_PWM[i], t0);
    logFor(STEP_OFF_MS, 0, t0);
  }
  setMotors(0);
  Serial.println(F("# done"));
}

void setup()
{
  Serial.begin(115200);
  analogReference(INTERNAL);    // 1.1 V reference, as in the stock firmware
  pinMode(AIN1, OUTPUT);
  pinMode(BIN1, OUTPUT);
  pinMode(PWMA_LEFT, OUTPUT);
  pinMode(PWMB_RIGHT, OUTPUT);
  pinMode(STBY_PIN, OUTPUT);
  pinMode(ENCODER_LEFT_A_PIN, INPUT);
  setMotors(0);
  digitalWrite(STBY_PIN, HIGH);
  attachInterrupt(digitalPinToInterrupt(ENCODER_LEFT_A_PIN), countEdge, CHANGE);
  Serial.println(F("# Motor_Step_Test ready. Commands: c z v g"));
}

void loop()
{
  if (Serial.available())
  {
    char c = Serial.read();
    if (c == 'c')
    {
      Serial.print(F("# edges: "));
      Serial.println(readCount());
    }
    else if (c == 'z')
    {
      noInterrupts();
      edge_count = 0;
      interrupts();
      Serial.println(F("# counter reset"));
    }
    else if (c == 'v')
    {
      Serial.print(F("# battery [V]: "));
      Serial.println(batteryVoltage());
    }
    else if (c == 'g')
    {
      runStepSequence();
    }
  }
}
