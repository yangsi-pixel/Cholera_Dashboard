const mongoose = require("mongoose");

const reportSchema = new mongoose.Schema(
  {
    date: {
      type: Date,
      required: true
    },
    pandemic: {
      type: String,
      required: true,
      trim: true
    },
    region: {
      type: String,
      required: true,
      trim: true
    },
    district: {
      type: String,
      required: true,
      trim: true
    },
    suspected: {
      type: Number,
      required: true,
      min: 0
    },
    confirmed: {
      type: Number,
      required: true,
      min: 0
    },
    deaths: {
      type: Number,
      required: true,
      min: 0
    },
    cfr: {
      type: Number,
      required: true,
      min: 0
    }
  },
  {
    timestamps: true
  }
);

module.exports = mongoose.model("Report", reportSchema);
