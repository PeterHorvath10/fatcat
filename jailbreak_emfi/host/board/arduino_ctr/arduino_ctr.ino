// Orin Nano reset: an Arduino on the carrier board's J14 header (pin 12 drives the power-button short).
// On the serial message msg it presses for delay_off ms (forced power-off), waits delay_between ms, then
// presses for delay_on ms (power on). host/emfi/reset.py sends the message (boards.orin.serial_message/serial_baud).

int pin = 12;
int incoming = 0;
int delay_between = 1000;
int delay_off = 20000;
int delay_on = 200;
int msg = 10;
void setup() {
  Serial.begin(9600);
  Serial.flush();
  pinMode(pin, OUTPUT);
  digitalWrite(pin, LOW);

}

void loop() {
  if (Serial.available() > 0) {
    incoming = (Serial.parseInt());

    if (incoming == msg) {
      Serial.write("ON\n");
      digitalWrite(pin, HIGH);
      delay(delay_off);      
      digitalWrite(pin, LOW);
      
      delay(delay_between);      

      digitalWrite(pin, HIGH);
      delay(delay_on);      
      digitalWrite(pin, LOW);
      incoming = 0;
      Serial.flush();

    }

  }
}
