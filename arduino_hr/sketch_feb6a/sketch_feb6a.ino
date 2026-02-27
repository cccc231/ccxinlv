#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"
MAX30105 particleSensor;
const byte RATE_SIZE = 4; 
byte rates[RATE_SIZE]; // 心率数组
byte rateSpot = 0;
long lastBeat = 0; // 最后记录心率的时间
float beatsPerMinute;
int beatAvg;
int Buzzer; 
int pin=8;
void setup()
{
  Serial.begin(115200);
  
  Serial.println("Initializing...");

  //   初始化传感器
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) //   默认使用I2C，400KHZ频率
  {
    Serial.println("MAX30105 was not found. Please check wiring/power. ");
    while (1);
  }
  Serial.println("Place your index finger on the sensor with steady pressure.");

  particleSensor.setup(); //使用默认设置配置传感器
  particleSensor.setPulseAmplitudeRed(0x0A); // 将红色LED拉低，表示传感器正在运行
}
void loop()
{
  long irValue = particleSensor.getIR();

  if (checkForBeat(irValue) == true)
  {
    //  感应到心率
    long delta = millis() - lastBeat;
    lastBeat = millis();
    beatsPerMinute = 60 / (delta / 1000.0);
    if (beatsPerMinute < 255 && beatsPerMinute > 20)
    {
      rates[rateSpot++] = (byte)beatsPerMinute; // 将此读数存储在数组中
      rateSpot %= RATE_SIZE; //Wrap variable  
      //Take average of readings  取读数的平均值
      beatAvg = 0;
      for (byte x = 0 ; x < RATE_SIZE ; x++)
        beatAvg += rates[x];
      beatAvg /= RATE_SIZE;
      if (beatsPerMinute > 70)
      {
       digitalWrite(8,HIGH) ;   //蜂鸣器响
       delay(1000);           //延时1000ms
       digitalWrite(8,LOW);   //蜂鸣器关闭
      }
    }
  }
  Serial.print("IR=");  
  Serial.print(irValue);
  Serial.print(", BPM="); 
  Serial.print(beatsPerMinute);
  Serial.print(", Avg BPM="); 
  Serial.print(beatAvg);
 
  if (irValue < 50000)
    Serial.print(" 未识别?");

  Serial.println();
}