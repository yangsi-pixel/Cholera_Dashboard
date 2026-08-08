const express = require("express");
const Report = require("../models/Report");

const router = express.Router();

// POST /api/reports: create and store a new pandemic report.
router.post("/", async (req, res, next) => {
  try {
    const { date, region, district, suspected, confirmed, deaths } = req.body;

    if (!date || !region || !district) {
      return res.status(400).json({
        message: "date, region, and district are required"
      });
    }

    const suspectedNum = Number(suspected) || 0;
    const confirmedNum = Number(confirmed) || 0;
    const deathsNum = Number(deaths) || 0;

    if (suspectedNum < 0 || confirmedNum < 0 || deathsNum < 0) {
      return res.status(400).json({
        message: "suspected, confirmed, and deaths must be non-negative"
      });
    }

    // CFR is always computed on the server to ensure trusted values.
    const cfr = confirmedNum === 0 ? 0 : Number(((deathsNum / confirmedNum) * 100).toFixed(2));

    const report = await Report.create({
      date,
      region,
      district,
      suspected: suspectedNum,
      confirmed: confirmedNum,
      deaths: deathsNum,
      cfr
    });

    return res.status(201).json({
      message: "Report saved successfully",
      data: report
    });
  } catch (error) {
    return next(error);
  }
});

// GET /api/reports: return all stored reports.
router.get("/", async (req, res, next) => {
  try {
    const reports = await Report.find().sort({ date: -1, createdAt: -1 });
    return res.status(200).json(reports);
  } catch (error) {
    return next(error);
  }
});

module.exports = router;
