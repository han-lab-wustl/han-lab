int ttlpin = A0; // pin that reads imaging ttl
int OptoOutpin = 10; // pin that outputs when to opto ttl
int count = 0; // timer for how many imaging frames have happened
int prevoptopin = 0; // previous frame used for next int
int ttldiff = 0; // diff used for check the start of an imaging ttl
unsigned long duration = 0; // how long are you stimulating for start with 0 opto (ms)
int interval = 1; // every n frames during opto it will send out a ttl
int OutpulseDur = 5; // how long will the out ttl be (ms)
unsigned long outttl = millis(); //initialize timer for ttl width 
int outdelay = 0; // how long do we wait to send out the ttl (ms) set to 0 for debugging
unsigned long start = millis(); //initialize timer for opto duration
int outstart = 0; // initialize state variable for timing the opto output pulse ttl

#include <Wire.h>

void setup() {
  // put your setup code here, to run once:
  Wire.begin(8);
 //Serial.begin(9600);
pinMode(ttlpin,INPUT);
pinMode(OptoOutpin,OUTPUT);
digitalWrite(OptoOutpin,LOW);
Wire.onReceive(receiveEvent);
}

void loop() {
  // put your main code here, to run repeatedly:
  int ttlbool = digitalRead(ttlpin); // read current imaging ttl state
 int ttldiff = ttlbool - prevoptopin; // calculate diff from previous frame to catch the start of a imaging ttl


//Serial.print(ttldiff);
//Serial.print(' ');
//Serial.print(ttlbool);
//Serial.print(' ');


  
if ((unsigned long)(millis()-start) < duration){ //during optogenetics
  if (ttldiff == 1) {// if start of ttl frame.  Reading and starting 
    count = count + 1; //add one imaging to the count
    if (count%interval == 0){ // if count is every 'interval' frame
      outstart = 1; // set up a timer for delaying your out ttl
      outttl = millis();
    }
  }
 
  if ((unsigned long)(millis()-outttl) > outdelay){ //after that delay
    if (outstart == 1){
      digitalWrite(OptoOutpin,HIGH); // start sending out   
      outstart = 2;
      }
    }
}


if (outstart == 2){ //make sure you turn it off after the duration of the out ttl
  if((unsigned long)(millis()-outttl)>=(OutpulseDur+outdelay)){
    digitalWrite(OptoOutpin,LOW);
    outstart = 0;
    }
  }

 //Serial.print("Variable1:");
 //Serial.print(digitalRead(OptoOutpin));
 //Serial.print(",");
prevoptopin = ttlbool; //save the previous state of imaging ttl for diff
//Serial.println(outstart);
}


void receiveEvent(){
  byte tempa[2]; // receive 2 bytes
  for (int i=0;i<2;i++){
   tempa[i] = Wire.read();
  }
  interval = (unsigned long)(tempa[0])+256*(unsigned long)(tempa[1]); //read the interval represented in 2 bytes
 // interval = tempa;
    tempa[1] = 0;
    tempa[2] = 0;
  for (int i=0;i<2;i++){
    tempa[i] = Wire.read();
  }
  duration = (unsigned long)(tempa[0])+256*(unsigned long)(tempa[1]); // read the duration represented in 2 bytes
  start = millis(); // reset the timer for starting a new opto every ping from matlab
   count = 0; //restart the frame count for every new opto ping.
 // Serial.print("Interval: ");
  //Serial.print(interval);
  //Serial.print(",");
  //Serial.print("Duration: ");
  // Serial.println(duration);
   
}

