const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const dotenv = require("dotenv");
const fs = require("fs/promises");
const path = require("path");

const reportsRouter = require("./routes/reports");

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// Allow requests from the CRA frontend running on localhost:3000.
app.use(
  cors({
    origin: "http://localhost:3000"
  })
);

app.use(express.json());

app.get("/", (req, res) => {
  res.json({ message: "Cholera Monitoring API is running." });
});

app.get("/api/regions-geojson", async (req, res, next) => {
  try {
    const geoJsonPath = path.join(
      __dirname,
      "..",
      "..",
      "..",
      "geoBoundaries-CMR-ADM1.geojson"
    );
    const geoJsonRaw = await fs.readFile(geoJsonPath, "utf-8");
    return res.status(200).json(JSON.parse(geoJsonRaw));
  } catch (error) {
    return next(error);
  }
});

app.use("/api/reports", reportsRouter);

// Centralized error handler for uncaught route-level errors.
app.use((err, req, res, next) => {
  console.error("Server error:", err);
  res.status(500).json({ message: "Internal server error" });
});

const startServer = async () => {
  try {
    // Example Atlas connection string is provided via process.env.MONGO_URI.
    await mongoose.connect(process.env.MONGO_URI);
    console.log("MongoDB connected successfully");

    app.listen(PORT, () => {
      console.log(`Server listening on port ${PORT}`);
    });
  } catch (error) {
    console.error("Failed to connect to MongoDB:", error.message);
    process.exit(1);
  }
};

startServer();
