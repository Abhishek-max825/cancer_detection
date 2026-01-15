import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, Activity, MousePointerClick } from 'lucide-react';
import ClinicalButton from '../components/ClinicalButton';
import GlassPanel from '../components/GlassPanel';

const LandingPage = () => {
    const navigate = useNavigate();

    return (
        <div className="min-h-screen pt-24 pb-12 px-6 flex flex-col items-center justify-center">
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
                className="text-center max-w-3xl space-y-8"
            >
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-medical-50 border border-medical-100 text-medical-700 text-sm font-medium">
                    <Activity className="w-4 h-4 animate-pulse" />
                    <span>Advanced Histopathological Analysis</span>
                </div>

                <h1 className="text-5xl md:text-7xl font-light text-clinical-900 leading-tight">
                    Precision AI for <br />
                    <span className="font-semibold text-medical-500">Tissue Diagnosis</span>
                </h1>

                <p className="text-xl text-clinical-500 max-w-2xl mx-auto font-light leading-relaxed">
                    Leveraging deep learning to detect metastatic cancer in histopathologic scans with high precision and explainable insights.
                </p>

                <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
                    <ClinicalButton
                        variant="primary"
                        onClick={() => navigate('/upload')}
                        icon={ArrowRight}
                        className="w-full sm:w-auto h-12 text-lg px-8 shadow-lg shadow-medical-500/25"
                    >
                        Launch Analyzer
                    </ClinicalButton>

                    <ClinicalButton
                        variant="ghost"
                        icon={MousePointerClick}
                        className="w-full sm:w-auto h-12"
                    >
                        View Demo
                    </ClinicalButton>
                </div>
            </motion.div>

            {/* Feature Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-24 w-full max-w-5xl">
                <FeatureCard
                    title="Clinical Accuracy"
                    description="Trained on thousands of biopsy samples for reliable detection."
                    delay={0.2}
                />
                <FeatureCard
                    title="Instant Analysis"
                    description="Real-time inference pipeline delivering results in seconds."
                    delay={0.4}
                />
                <FeatureCard
                    title="Explainable AI"
                    description="Grad-CAM visualizations to highlight regions of interest."
                    delay={0.6}
                />
            </div>

            <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1, duration: 1 }}
                className="mt-24 text-xs text-clinical-400 text-center max-w-md"
            >
                FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY. <br />
                NOT INTENDED FOR CLINICAL DIAGNOSIS WITHOUT PATHOLOGIST REVIEW.
            </motion.p>
        </div>
    );
};

const FeatureCard = ({ title, description, delay }) => (
    <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay, duration: 0.6 }}
    >
        <GlassPanel hoverEffect className="h-full border-t-4 border-t-medical-500/0 hover:border-t-medical-500 transition-all">
            <h3 className="text-lg font-semibold mb-2 text-clinical-900">{title}</h3>
            <p className="text-clinical-500 leading-relaxed text-sm">{description}</p>
        </GlassPanel>
    </motion.div>
);

export default LandingPage;
