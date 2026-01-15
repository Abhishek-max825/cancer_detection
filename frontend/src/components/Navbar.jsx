import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, ShieldCheck, Info } from 'lucide-react';

const Navbar = () => {
    const location = useLocation();

    const isActive = (path) => location.pathname === path;

    return (
        <nav className="fixed top-0 w-full z-50 px-6 py-4">
            <div className="max-w-7xl mx-auto">
                <div className="glass-panel px-6 py-3 flex items-center justify-between">
                    {/* Logo Area */}
                    <Link to="/" className="flex items-center gap-2 group">
                        <div className="w-8 h-8 rounded-lg bg-medical-50 text-medical-600 flex items-center justify-center group-hover:bg-medical-500 group-hover:text-white transition-colors duration-300">
                            <Activity className="w-5 h-5" />
                        </div>
                        <span className="font-display font-semibold text-lg tracking-tight text-clinical-900">
                            Histo<span className="text-medical-500">Scan</span> AI
                        </span>
                    </Link>

                    {/* Navigation Links */}
                    <div className="hidden md:flex items-center gap-8">
                        <NavLink to="/" icon={Info} active={isActive('/')}>About</NavLink>
                        <NavLink to="/analyze" icon={ShieldCheck} active={isActive('/analyze')}>Scan Image</NavLink>
                    </div>

                    {/* Status Indicator */}
                    <div className="flex items-center gap-2 text-xs font-medium text-emerald-600 bg-emerald-50 px-3 py-1.5 rounded-full border border-emerald-100">
                        <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                        </span>
                        System Online
                    </div>
                </div>
            </div>
        </nav>
    );
};

const NavLink = ({ to, children, active, icon: Icon }) => (
    <Link
        to={to}
        className={`flex items-center gap-2 text-sm font-medium transition-colors duration-200 ${active ? 'text-medical-600' : 'text-clinical-500 hover:text-medical-600'
            }`}
    >
        {Icon && <Icon className="w-4 h-4" />}
        {children}
    </Link>
);

export default Navbar;
