/*
 * Tuning.h - live gain tuning and telemetry for the stock ELEGOO Tumbller firmware
 * (Tumbller balance course, notebook 4)
 *
 * What it adds
 * ------------
 *  - Telemetry: every TELEMETRY_PERIOD_MS one CSV line
 *        ms,angle,gyro,pwm,speed,integral
 *    angle [deg], gyro [deg/s], pwm [-255..255] (left motor),
 *    speed [encoder counts per 40 ms, filtered], integral [counts]
 *  - Commands (Serial Monitor, 9600 baud, line ending "Newline"):
 *        P55.0   kp_balance          S10.0   kp_speed
 *        D0.75   kd_balance          I0.26   ki_speed
 *        Z-1.2   angle_zero [deg]    T0 / T1 telemetry off / on
 *        ?       print the current gains
 *    The stock one-letter commands (f b l i s 0..9 * #) keep working.
 *
 * Lines that are not data start with '#', so a logger can skip them.
 * Gains changed here live in RAM only: after a reset the values from
 * BalanceCar.h are back. Write the ones you like into BalanceCar.h.
 *
 * Must be included AFTER BalanceCar.h (it uses that file's variables).
 */

#define TELEMETRY_PERIOD_MS 50

char tuning_buf[16];              // characters of the command being received
unsigned char tuning_len = 0;
bool telemetry_on = true;
unsigned long telemetry_time = 0;

void tuningPrintGains()
{
  Serial.print(F("# ms="));       // time stamp, so a log shows WHEN a gain changed
  Serial.print(millis());
  Serial.print(F(" kp_balance="));
  Serial.print(kp_balance);
  Serial.print(F(" kd_balance="));
  Serial.print(kd_balance);
  Serial.print(F(" kp_speed="));
  Serial.print(kp_speed);
  Serial.print(F(" ki_speed="));
  Serial.print(ki_speed);
  Serial.print(F(" angle_zero="));
  Serial.println(angle_zero);
}

void tuningApply()                // a complete command line is in tuning_buf
{
  tuning_buf[tuning_len] = '\0';
  double value = atof(tuning_buf + 1);
  noInterrupts();                 // balanceCar() must not run while a gain is half written
  switch (tuning_buf[0])
  {
  case 'P':
    kp_balance = value;
    break;
  case 'D':
    kd_balance = value;
    break;
  case 'S':
    kp_speed = value;
    break;
  case 'I':
    ki_speed = value;
    car_speed_integeral = 0;      // avoid a jump from the old integral
    break;
  case 'Z':
    angle_zero = value;
    break;
  case 'T':
    telemetry_on = (value != 0);
    break;
  default:
    break;
  }
  interrupts();
  tuning_len = 0;
  tuningPrintGains();
}

// Replaces getBluetoothData() of Command.h: same behaviour for the stock
// one-letter commands, plus the tuning commands above.
bool tuningSerial()
{
  while (Serial.available())
  {
    char c = Serial.read();
    if (tuning_len > 0)           // we are inside a tuning command
    {
      if (c == '\n' || c == '\r')
      {
        tuningApply();
      }
      else if (tuning_len < sizeof(tuning_buf) - 1)
      {
        tuning_buf[tuning_len++] = c;
      }
      continue;
    }
    if (c == '?')
    {
      tuningPrintGains();
      continue;
    }
    if (c >= 'A' && c <= 'Z')     // upper-case letter starts a tuning command
    {
      tuning_buf[tuning_len++] = c;
      continue;
    }
    if (c != '\0' && c != '\n' && c != '\r')   // stock behaviour from here on
    {
      key_value = c;
      if (c == 'f' || c == 'b' || c == 'l' || c == 'i')
      {
        function_mode = BLUETOOTH;
      }
      if (c == 's')
      {
        function_mode = IDLE;
      }
      return true;
    }
  }
  return false;
}

void tuningTelemetry()
{
  if (!telemetry_on || millis() - telemetry_time < TELEMETRY_PERIOD_MS)
  {
    return;
  }
  telemetry_time = millis();
  noInterrupts();                 // take a consistent snapshot
  float angle = kalmanfilter_angle;
  float gyro = kalmanfilter.Gyro_x;
  double pwm = pwm_left;
  double speed = speed_filter;
  double integral = car_speed_integeral;
  interrupts();
  Serial.print(telemetry_time);
  Serial.print(',');
  Serial.print(angle, 2);
  Serial.print(',');
  Serial.print(gyro, 1);
  Serial.print(',');
  Serial.print((int)pwm);
  Serial.print(',');
  Serial.print(speed, 1);
  Serial.print(',');
  Serial.println((long)integral);
}
