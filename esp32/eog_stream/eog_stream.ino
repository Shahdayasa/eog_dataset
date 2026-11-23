// EOG streamer: samples the BioAmp at 1000 Hz and POSTs 1 s batches to the server.
//
// Two tasks so network delays never stop sampling:
//   samplingTask (core 1, high priority) - reads the ADC every 1 ms, fills 1000-sample batches
//   loop()       (network)               - takes batches from a queue and POSTs them
// The queue holds QUEUE_DEPTH seconds. If the network is slower than that, batches are
// dropped; the sequence number lets the server see the gap and restart its window.

#include <Arduino.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>

// ── Fill these in ────────────────────────────────────────────────────────────
const char *WIFI_SSID  = "YOUR_WIFI_NAME";                      // 2.4 GHz network
const char *WIFI_PASS  = "YOUR_WIFI_PASSWORD";
const char *SERVER_URL = "http://YOUR_LAPTOP_IP:8000/ingest";   // laptop/server IP (ipconfig)

const char *API_KEY    = "CHANGE_ME";                           // same as EOG_API_KEY on the server
const char *DEVICE_ID  = "esp32-01";

// ── Signal configuration ─────────────────────────────────────────────────────
const int ADC_PIN     = 34;     // must be an ADC1 pin (GPIO32-39); ADC2 does not work with WiFi on
const int FS_HZ       = 1000;
const int BATCH_N     = 1000;   // samples per POST = 1 s
const int QUEUE_DEPTH = 8;      // seconds of buffering while the network is slow

static_assert(configTICK_RATE_HZ == 1000, "needs a 1 ms FreeRTOS tick for 1000 Hz sampling");

struct Batch {
  uint32_t seq;                 // index of the first sample since boot
  uint16_t mv[BATCH_N];         // ADC input in mV (calibrated by analogReadMilliVolts)
};

QueueHandle_t batchQueue;
volatile uint32_t droppedBatches = 0;

WiFiClientSecure secureClient;
WiFiClient plainClient;
HTTPClient http;

void samplingTask(void *) {
  static Batch batch;
  uint32_t seq = 0;
  int n = 0;
  TickType_t lastWake = xTaskGetTickCount();
  for (;;) {
    vTaskDelayUntil(&lastWake, 1);              // exactly every 1 ms, no drift
    if (n == 0) batch.seq = seq;
    batch.mv[n++] = analogReadMilliVolts(ADC_PIN);
    seq++;
    if (n == BATCH_N) {
      if (xQueueSend(batchQueue, &batch, 0) != pdTRUE) droppedBatches++;
      n = 0;
    }
  }
}

void ensureWifi() {
  if (WiFi.status() == WL_CONNECTED) return;
  Serial.print("WiFi connecting");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print('.');
  }
  Serial.print(" connected, IP ");
  Serial.println(WiFi.localIP());
}

bool postBatch(const Batch &b) {
  bool https = strncmp(SERVER_URL, "https://", 8) == 0;
  bool ok = https ? http.begin(secureClient, SERVER_URL) : http.begin(plainClient, SERVER_URL);
  if (!ok) return false;
  http.addHeader("Content-Type", "application/octet-stream");
  http.addHeader("X-API-Key", API_KEY);
  http.addHeader("X-Device-Id", DEVICE_ID);
  http.addHeader("X-Seq", String(b.seq));
  // ESP32 is little-endian, so the uint16 array is already in the wire format.
  int code = http.POST((uint8_t *)b.mv, sizeof(b.mv));
  String reply = code > 0 ? http.getString() : http.errorToString(code);
  http.end();                                   // keeps the connection open (setReuse)

  if (code == 200) {
    if (reply.indexOf("\"prediction\":\"") >= 0) Serial.println(reply);   // a new 10 s result
    return true;
  }
  Serial.printf("POST failed (%d): %s\n", code, reply.c_str());
  return false;
}

void setup() {
  // TEMPORARY: the USB supply browns out when WiFi starts. Remove this line once the
  // board has proper power (better cable, or 470-1000 uF capacitor between 5V and GND).
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  Serial.begin(115200);
  delay(1000);                                  // give the Serial Monitor time to attach
  Serial.println("\n[1] booted");
  Serial.flush();                               // make sure each message is fully sent before the next step
  analogReadResolution(12);
  analogSetPinAttenuation(ADC_PIN, ADC_11db);   // ~0-3.1 V range
  Serial.println("[2] ADC ready, starting WiFi");
  Serial.flush();

  WiFi.mode(WIFI_STA);
  WiFi.setTxPower(WIFI_POWER_8_5dBm);           // lower current spikes (avoids brownout on weak USB); phone is close by
  WiFi.setSleep(false);                         // lower latency
  Serial.println("[3] WiFi radio on");
  Serial.flush();
  ensureWifi();

  // TODO for production: replace setInsecure() with setCACert(<server root CA>).
  secureClient.setInsecure();
  http.setReuse(true);
  http.setTimeout(5000);

  batchQueue = xQueueCreate(QUEUE_DEPTH, sizeof(Batch));
  if (batchQueue == NULL) {
    Serial.println("ERROR: not enough memory for the batch queue");
    while (true) delay(1000);
  }
  xTaskCreatePinnedToCore(samplingTask, "sampling", 4096, NULL, 10, NULL, 1);
  Serial.println("[4] sampling started, sending to server");
}

void loop() {
  static Batch b;
  static uint32_t reportedDrops = 0;
  if (xQueueReceive(batchQueue, &b, portMAX_DELAY) != pdTRUE) return;

  ensureWifi();
  // One retry; a lost batch just makes the server restart its 10 s window.
  if (!postBatch(b) && !postBatch(b)) Serial.printf("batch %u lost\n", b.seq);

  if (droppedBatches != reportedDrops) {
    reportedDrops = droppedBatches;
    Serial.printf("WARNING: %u batches dropped (network too slow)\n", reportedDrops);
  }
}
