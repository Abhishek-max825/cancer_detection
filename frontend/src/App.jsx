import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import LandingPage from './pages/LandingPage';
import UploadPage from './pages/UploadPage';
import ResultPage from './pages/ResultPage';
import './index.css';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-clinical-50 text-clinical-900 font-body">
        <Navbar />
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/result" element={<ResultPage />} />
          {/* Redirect old route for compatibility */}
          <Route path="/analyze" element={<UploadPage />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
