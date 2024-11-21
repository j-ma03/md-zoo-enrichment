#include <Arduino_LSM6DS3.h> // Include the library
#include <Ethernet.h>
#include <SPI.h>
#include <SD.h>

const int chipSelect = 10;
File dataFile;

void setup() {

  Serial.begin(1000000); // Initialize serial communication
  SD.begin(chipSelect); //pin 10
  dataFile = SD.open("ZooData.txt",FILE_WRITE);

  if (!IMU.begin()) { // Initialize IMU

    Serial.println("IMU initialization failed!"); 

    while (1); // Stop if initialization fails

  }

}


void loop() {

  float ax, ay, az; // Accelerometer values

  float gx, gy, gz; // Gyroscope values



  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) { // Check if data is available

    IMU.readAcceleration(ax, ay, az); // Read accelerometer data

    IMU.readGyroscope(gx, gy, gz); // Read gyroscope data

    // Write data to the SD card
    dataFile = SD.open("ZOODATA.txt", FILE_WRITE); // Open file in append mode
    if (dataFile) {
      dataFile.print("Acc X: "); dataFile.print(ax);
      dataFile.print(", Acc Y: "); dataFile.print(ay);
      dataFile.print(", Acc Z: "); dataFile.println(az);
      dataFile.close(); // Close the file to save data
    } else {
      Serial.println("Error opening ZooData.txt!");
    }

    Serial.print("Acc X: "); Serial.print(ax); 

    Serial.print(", Acc Y: "); Serial.print(ay); 

    Serial.print(", Acc Z: "); Serial.println(az);



    //Serial.print("Gyro X: "); Serial.print(gx); 

   // Serial.print(", Gyro Y: "); Serial.print(gy); 

    //Serial.print(", Gyro Z: "); Serial.println(gz);

  }
  // Serial.print("IP address: "); 

  //       Serial.println(Ethernet.localIP()); // Print the IP address

  delay(100); // Delay between readings

}
