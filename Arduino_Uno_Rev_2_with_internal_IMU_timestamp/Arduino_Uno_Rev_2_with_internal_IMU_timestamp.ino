#include <Arduino_LSM6DS3.h> // Include the library
#include "RTClib.h"
#include <Ethernet.h>
#include <SPI.h>
#include <SD.h>

const int chipSelect = 10;
File dataFile;
RTC_DS3231 rtc;

void setup() {

  Serial.begin(9600); // Initialize serial communication
  // SD.begin(chipSelect); //pin 10
  // dataFile = SD.open("ZooData.txt",FILE_WRITE);

  // Initialize SD card
  if (!SD.begin(chipSelect)) { 
    Serial.println("SD card initialization failed!");
    while (1); // Stop if SD initialization fails
  }

  // Open the file in write mode to overwrite and immediately close it
  dataFile = SD.open("ZOODATA.txt", O_WRITE | O_CREAT | O_TRUNC);
  if (dataFile) {
    dataFile.close(); // Close it to overwrite content
  } else {
    Serial.println("Error creating ZOODATA.txt!");
  }

  // Initialize IMU
  if (!IMU.begin()) { 

    Serial.println("IMU initialization failed!"); 

    while (1); // Stop if initialization fails

  }

  // Initialize clock
  #ifndef ESP8266
  while (!Serial); // wait for serial port to connect. Needed for native USB
  #endif

  if (! rtc.begin()) {
    Serial.println("Couldn't find RTC");
    Serial.flush();
    while (1) delay(10);
  }

  if (rtc.lostPower()) {
    Serial.println("RTC lost power, let's set the time!");
    // When time needs to be set on a new device, or after a power loss, the
    // following line sets the RTC to the date & time this sketch was compiled
    rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
    // This line sets the RTC with an explicit date & time, for example to set
    // January 21, 2014 at 3am you would call:
    // rtc.adjust(DateTime(2014, 1, 21, 3, 0, 0));
  }

  // When time needs to be re-set on a previously configured device, the
  // following line sets the RTC to the date & time this sketch was compiled
  // rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
  // This line sets the RTC with an explicit date & time, for example to set
  // January 21, 2014 at 3am you would call:
  // rtc.adjust(DateTime(2014, 1, 21, 3, 0, 0));

}


void loop() {

  float ax, ay, az; // Accelerometer values

  float gx, gy, gz; // Gyroscope values



  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) { // Check if data is available

    IMU.readAcceleration(ax, ay, az); // Read accelerometer data

    IMU.readGyroscope(gx, gy, gz); // Read gyroscope data

    DateTime now = rtc.now(); // get current time

    // Write data to the SD card
    dataFile = SD.open("ZOODATA.txt", FILE_WRITE); // Open file in append mode
    if (dataFile) {
      // write time
      dataFile.print(now.year(), DEC);
      dataFile.print(' ');
      dataFile.print(now.month(), DEC);
      dataFile.print(' ');
      dataFile.print(now.day(), DEC);
      dataFile.print(" ");
      dataFile.print(now.hour(), DEC);
      dataFile.print(' ');
      dataFile.print(now.minute(), DEC);
      dataFile.print(' ');
      dataFile.print(now.second(), DEC);

      // write acceleration
      dataFile.print(" ");
      dataFile.print(ax);
      dataFile.print(" ");
      dataFile.print(ay);
      dataFile.print(" ");
      dataFile.println(az);
      dataFile.close(); // Close the file to save data
    } else {
      Serial.println("Error opening ZOODATA.txt!");
    }

    // print time
    Serial.print(now.year(), DEC);
    Serial.print('/');
    Serial.print(now.month(), DEC);
    Serial.print('/');
    Serial.print(now.day(), DEC);
    Serial.print(" ");
    Serial.print(now.hour(), DEC);
    Serial.print(':');
    Serial.print(now.minute(), DEC);
    Serial.print(':');
    Serial.print(now.second(), DEC);

    // print acceleration
    Serial.print(" ");

    Serial.print(ax); 
    Serial.print(" ");

    Serial.print(ay); 
    Serial.print(" ");

    Serial.println(az);



    //Serial.print("Gyro X: "); Serial.print(gx); 

   // Serial.print(", Gyro Y: "); Serial.print(gy); 

    //Serial.print(", Gyro Z: "); Serial.println(gz);

  }
  // Serial.print("IP address: "); 

  //       Serial.println(Ethernet.localIP()); // Print the IP address

  delay(100); // Delay between readings

}
