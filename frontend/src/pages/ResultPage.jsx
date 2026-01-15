import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { RotateCcw, ArrowLeft, FileText } from 'lucide-react';
import ClinicalButton from '../components/ClinicalButton';
import HeatmapViewer from '../components/HeatmapViewer';
import { generateReport } from '../utils/ReportGenerator';

const ResultPage = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const { result, preview } = location.state || {};

    // Redirect to upload if no result data is present
    React.useEffect(() => {
        if (!result) {
            navigate('/upload');
        }
    }, [result, navigate]);

    if (!result) return null;

    return (
        <div className="min-h-screen pt-24 pb-12 px-6 flex flex-col items-center">
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="w-full max-w-4xl space-y-8"
            >
                <div className="flex items-center justify-between">
                    <button
                        onClick={() => navigate('/upload')}
                        className="flex items-center gap-2 text-clinical-500 hover:text-clinical-700 transition-colors"
                    >
                        <ArrowLeft className="w-4 h-4" />
                        <span>Back to Upload</span>
                    </button>
                    <h2 className="text-2xl font-light text-clinical-900">Analysis Results</h2>
                    <div className="w-24"></div> {/* Spacer for centering */}
                </div>

                <HeatmapViewer
                    originalImage={result.original_base64 || preview}
                    heatmapImage={result.heatmap_base64}
                    prediction={result.prediction}
                    confidence={result.confidence}
                />

                <div className="mt-12 flex justify-center gap-4">
                    <ClinicalButton
                        variant="secondary"
                        onClick={() => navigate('/upload')}
                        icon={RotateCcw}
                        className="shadow-lg shadow-clinical-500/10"
                    >
                        Analyze Another Sample
                    </ClinicalButton>

                    <ClinicalButton
                        variant="primary"
                        onClick={() => generateReport(result, preview)}
                        icon={FileText}
                        className="shadow-lg shadow-medical-500/20"
                    >
                        Download Report
                    </ClinicalButton>
                </div>
            </motion.div>
        </div>
    );
};

export default ResultPage;
