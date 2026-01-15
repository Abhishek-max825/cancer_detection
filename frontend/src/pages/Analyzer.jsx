import React, { useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import { AlertCircle, CheckCircle2, RotateCcw } from 'lucide-react';
import ClinicalButton from '../components/ClinicalButton';
import UploadZone from '../components/UploadZone';
import HeatmapViewer from '../components/HeatmapViewer';
import GlassPanel from '../components/GlassPanel';

const Analyzer = () => {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleFileSelect = (selectedFile) => {
        setFile(selectedFile);
        setPreview(URL.createObjectURL(selectedFile));
        setResult(null);
        setError(null);
    };

    const handleClear = () => {
        setFile(null);
        setPreview(null);
        setResult(null);
        setError(null);
    };

    const handleAnalyze = async () => {
        if (!file) return;

        setLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', file);

        try {
            // Assuming backend runs on port 8000
            const response = await axios.post('http://localhost:8000/predict', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setResult(response.data);
        } catch (err) {
            console.error(err);
            setError('Analysis failed. Please ensure the backend is running and try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen pt-24 pb-12 px-6 flex flex-col items-center">
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="w-full max-w-4xl space-y-8"
            >
                <div className="text-center space-y-2">
                    <h2 className="text-3xl font-light text-clinical-900">Tissue Analyzer</h2>
                    <p className="text-clinical-500">Upload a histological slide for instant malignancy verification.</p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
                    {/* Left Column: Upload & Controls */}
                    <div className="space-y-6">
                        <GlassPanel className="bg-white/50">
                            <UploadZone
                                onFileSelect={handleFileSelect}
                                selectedFile={file}
                                onClear={handleClear}
                            />

                            <div className="mt-6 flex justify-end">
                                <ClinicalButton
                                    onClick={handleAnalyze}
                                    disabled={!file || loading || result}
                                    isLoading={loading}
                                    className="w-full sm:w-auto"
                                >
                                    {result ? 'Analysis Complete' : 'Run Analysis'}
                                </ClinicalButton>
                            </div>
                        </GlassPanel>

                        {error && (
                            <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                className="bg-red-50 border border-red-100 text-red-600 p-4 rounded-xl flex items-center gap-3"
                            >
                                <AlertCircle className="w-5 h-5 shrink-0" />
                                <p className="text-sm">{error}</p>
                            </motion.div>
                        )}
                    </div>

                    {/* Right Column: Results */}
                    <div className="space-y-6">
                        {!result && !loading && (
                            <div className="h-64 flex flex-col items-center justify-center text-clinical-400 border-2 border-dashed border-clinical-100 rounded-2xl">
                                <p>Results will appear here</p>
                            </div>
                        )}

                        {loading && (
                            <div className="h-64 flex flex-col items-center justify-center space-y-4">
                                <div className="relative w-16 h-16">
                                    <div className="absolute inset-0 rounded-full border-4 border-medical-100"></div>
                                    <div className="absolute inset-0 rounded-full border-4 border-medical-500 border-t-transparent animate-spin"></div>
                                </div>
                                <p className="text-medical-600 font-medium animate-pulse">Running Inference Models...</p>
                            </div>
                        )}

                        {result && (
                            <motion.div
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                            >
                                <HeatmapViewer
                                    originalImage={result.original_base64 || preview}
                                    heatmapImage={result.heatmap_base64}
                                    prediction={result.prediction}
                                    confidence={result.confidence}
                                />

                                <div className="mt-6 flex justify-center">
                                    <ClinicalButton
                                        variant="secondary"
                                        onClick={handleClear}
                                        icon={RotateCcw}
                                    >
                                        Analyze Another Sample
                                    </ClinicalButton>
                                </div>
                            </motion.div>
                        )}
                    </div>
                </div>
            </motion.div>
        </div>
    );
};

export default Analyzer;
