import React, { useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import { AlertCircle, ArrowLeft } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import ClinicalButton from '../components/ClinicalButton';
import UploadZone from '../components/UploadZone';
import GlassPanel from '../components/GlassPanel';

const UploadPage = () => {
    const navigate = useNavigate();
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleFileSelect = (selectedFile) => {
        setFile(selectedFile);
        setPreview(URL.createObjectURL(selectedFile));
        setError(null);
    };

    const handleClear = () => {
        setFile(null);
        setPreview(null);
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

            // Navigate to result page with data
            navigate('/result', {
                state: {
                    result: response.data,
                    preview: preview
                }
            });
        } catch (err) {
            console.error(err);
            setError('Analysis failed. Please ensure the backend is running and try again.');
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen pt-24 pb-12 px-6 flex flex-col items-center">
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="w-full max-w-2xl space-y-8"
            >
                <div className="flex items-center justify-between">
                    <button
                        onClick={() => navigate('/')}
                        className="flex items-center gap-2 text-clinical-500 hover:text-clinical-700 transition-colors"
                    >
                        <ArrowLeft className="w-4 h-4" />
                        <span>Back to Home</span>
                    </button>
                    <div className="w-24"></div> {/* Spacer */}
                </div>

                <div className="text-center space-y-2">
                    <h2 className="text-3xl font-light text-clinical-900">Tissue Analyzer</h2>
                    <p className="text-clinical-500">Upload a histological slide for instant malignancy verification.</p>
                </div>

                <div className="space-y-6">
                    <GlassPanel className="bg-clinical-100/50 relative overflow-hidden">
                        <UploadZone
                            onFileSelect={handleFileSelect}
                            selectedFile={file}
                            onClear={handleClear}
                        />

                        {/* Scanner Animation Overlay */}
                        {loading && preview && (
                            <div className="absolute inset-0 z-50 bg-clinical-900/40 backdrop-blur-sm flex items-center justify-center rounded-2xl overflow-hidden">
                                {/* Moving Laser Line */}
                                <motion.div
                                    initial={{ top: "0%" }}
                                    animate={{ top: "100%" }}
                                    transition={{
                                        duration: 2,
                                        repeat: Infinity,
                                        ease: "linear"
                                    }}
                                    className="absolute left-0 right-0 h-1 bg-medical-400 shadow-[0_0_20px_rgba(45,212,191,0.8)] z-20"
                                />

                                {/* Scanning Grid Effect */}
                                <div className="absolute inset-0 bg-[linear-gradient(rgba(6,182,212,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(6,182,212,0.1)_1px,transparent_1px)] bg-[size:40px_40px] z-10"></div>

                                <div className="relative z-30 bg-clinical-900/80 px-6 py-3 rounded-full border border-medical-500/30 backdrop-blur-md flex items-center gap-3 shadow-xl">
                                    <div className="w-2 h-2 rounded-full bg-medical-500 animate-pulse" />
                                    <span className="text-medical-100 font-mono tracking-wider text-sm">SCANNING TISSUE...</span>
                                </div>
                            </div>
                        )}

                        <div className="mt-6 flex justify-end">
                            <ClinicalButton
                                onClick={handleAnalyze}
                                disabled={!file || loading}
                                isLoading={loading}
                                className="w-full sm:w-auto"
                            >
                                {loading ? 'Processing...' : 'Run Analysis'}
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
            </motion.div>
        </div>
    );
};

export default UploadPage;
