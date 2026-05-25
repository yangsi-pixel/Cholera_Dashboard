import { render, screen } from '@testing-library/react';
import App from './App';

test('renders cholera admin title', () => {
  render(<App />);
  const headingElement = screen.getByText(/cholera case data entry - cameroon/i);
  expect(headingElement).toBeInTheDocument();
});
