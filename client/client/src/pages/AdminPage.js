import { useMemo, useState } from "react";
import axios from "axios";
import cameroonDistricts, { regionOptions } from "../data/cameroonDistricts";
import "../styles/AdminPage.css";

const initialFormState = {
  date: "",
  pandemic: "",
  region: "",
  district: "",
  suspected: "",
  confirmed: "",
  deaths: ""
};

const pandemicOptions = ["Cholera", "COVID-19", "Measles", "Mpox", "Influenza"];

function AdminPage() {
  const [formData, setFormData] = useState(initialFormState);
  const [loading, setLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  const cfr = useMemo(() => {
    const confirmed = Number(formData.confirmed) || 0;
    const deaths = Number(formData.deaths) || 0;

    if (confirmed === 0) {
      return 0;
    }

    return Number(((deaths / confirmed) * 100).toFixed(2));
  }, [formData.confirmed, formData.deaths]);

  const handleChange = (event) => {
    const { name, value } = event.target;

    setFormData((prev) => {
      if (name === "region") {
        return { ...prev, region: value, district: "" };
      }

      return { ...prev, [name]: value };
    });

    setSuccessMessage("");
    setErrorMessage("");
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setSuccessMessage("");
    setErrorMessage("");

    const payload = {
      date: formData.date,
      pandemic: formData.pandemic,
      region: formData.region,
      district: formData.district,
      suspected: Number(formData.suspected) || 0,
      confirmed: Number(formData.confirmed) || 0,
      deaths: Number(formData.deaths) || 0,
      cfr
    };

    try {
      await axios.post("http://localhost:5000/api/reports", payload);
      setSuccessMessage("Report submitted successfully.");
      setFormData(initialFormState);
    } catch (error) {
      setErrorMessage(
        error.response?.data?.message ||
          "Submission failed. Please verify the server and try again."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="admin-page">
      <section className="admin-card" aria-labelledby="admin-page-title">
        <h1 id="admin-page-title">Pandemic Case Data Entry - Cameroon</h1>
        <p className="admin-subtitle">
          Use this form to submit daily district-level pandemic surveillance reports.
        </p>

        <form className="admin-form" onSubmit={handleSubmit}>
          <label htmlFor="date">Report Date</label>
          <input
            id="date"
            name="date"
            type="date"
            value={formData.date}
            onChange={handleChange}
            required
          />

          <label htmlFor="pandemic">Pandemic</label>
          <select
            id="pandemic"
            name="pandemic"
            value={formData.pandemic}
            onChange={handleChange}
            required
          >
            <option value="">Select a pandemic</option>
            {pandemicOptions.map((pandemic) => (
              <option key={pandemic} value={pandemic}>
                {pandemic}
              </option>
            ))}
          </select>

          <label htmlFor="region">Region</label>
          <select
            id="region"
            name="region"
            value={formData.region}
            onChange={handleChange}
            required
          >
            <option value="">Select a region</option>
            {regionOptions.map((region) => (
              <option key={region} value={region}>
                {region}
              </option>
            ))}
          </select>

          <label htmlFor="district">District</label>
          <input
            id="district"
            name="district"
            type="text"
            placeholder="Enter district name"
            value={formData.district}
            onChange={handleChange}
            required
          />

          <label htmlFor="suspected">Suspected Cases</label>
          <input
            id="suspected"
            name="suspected"
            type="number"
            min="0"
            value={formData.suspected}
            onChange={handleChange}
            required
          />

          <label htmlFor="confirmed">Confirmed Cases</label>
          <input
            id="confirmed"
            name="confirmed"
            type="number"
            min="0"
            value={formData.confirmed}
            onChange={handleChange}
            required
          />

          <label htmlFor="deaths">Deaths</label>
          <input
            id="deaths"
            name="deaths"
            type="number"
            min="0"
            value={formData.deaths}
            onChange={handleChange}
            required
          />

          <label htmlFor="cfr">CFR (%)</label>
          <input id="cfr" name="cfr" type="text" value={cfr} readOnly />

          <button type="submit" disabled={loading}>
            {loading ? "Submitting..." : "Submit Report"}
          </button>
        </form>

        {successMessage && (
          <p className="status-message success" role="status">
            {successMessage}
          </p>
        )}

        {errorMessage && (
          <p className="status-message error" role="alert">
            {errorMessage}
          </p>
        )}
      </section>
    </main>
  );
}

export default AdminPage;
