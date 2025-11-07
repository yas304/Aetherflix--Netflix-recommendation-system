import { Link } from 'react-router-dom'
import { FaBrain, FaChartLine, FaRobot, FaDatabase, FaArrowRight } from 'react-icons/fa'
import { SiTensorflow, SiPython, SiScikitlearn } from 'react-icons/si'

export default function Landing() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-netflix-black via-gray-900 to-netflix-black">
      {/* Hero Section */}
      <div className="relative">
        {/* Hero Content */}
        <div className="container mx-auto px-6 py-20">
          <div className="max-w-4xl mx-auto text-center">
            <div className="inline-block px-4 py-2 bg-netflix-red/20 rounded-full mb-6">
              <span className="text-netflix-red font-semibold">🤖 AI-Powered Machine Learning System</span>
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold mb-6 slide-up bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
              Movie Content Classification & Recommendation Engine
            </h1>
            
            <p className="text-xl text-gray-400 mb-8 max-w-3xl mx-auto slide-up" style={{ animationDelay: '0.2s' }}>
              Advanced machine learning system using <span className="text-netflix-red">XGBoost</span>, <span className="text-netflix-red">Transformers</span>, and <span className="text-netflix-red">FAISS</span> for intelligent movie classification and content-based recommendations
            </p>

            <div className="flex flex-wrap gap-4 justify-center mb-12 slide-up" style={{ animationDelay: '0.4s' }}>
              <Link to="/signup" className="btn-primary flex items-center gap-2 text-lg px-8 py-3">
                <FaRobot /> Try ML Models <FaArrowRight className="text-sm" />
              </Link>
              <a href="#features" className="btn-secondary flex items-center gap-2 text-lg px-8 py-3">
                <FaChartLine /> View Features
              </a>
            </div>

            {/* Tech Stack */}
            <div className="flex flex-wrap gap-6 justify-center items-center text-gray-500 slide-up" style={{ animationDelay: '0.6s' }}>
              <div className="flex items-center gap-2">
                <SiPython className="text-2xl" />
                <span>Python</span>
              </div>
              <div className="flex items-center gap-2">
                <SiTensorflow className="text-2xl" />
                <span>Transformers</span>
              </div>
              <div className="flex items-center gap-2">
                <SiScikitlearn className="text-2xl" />
                <span>XGBoost</span>
              </div>
              <div className="flex items-center gap-2">
                <FaDatabase className="text-2xl" />
                <span>FAISS Vector DB</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div id="features" className="py-20 px-12 bg-netflix-darkGray">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl font-bold mb-4 text-center">
            AI-Powered <span className="text-netflix-red">Features</span>
          </h2>
          <p className="text-gray-400 text-center mb-12 max-w-2xl mx-auto">
            Explore the machine learning capabilities powering this content intelligence platform
          </p>
          
          <div className="grid md:grid-cols-3 gap-8">
            <FeatureCard
              title="Content Classification"
              description="XGBoost-powered multi-label classification predicting genres, ratings, and content types with 95%+ accuracy on the Netflix dataset"
              icon={<FaBrain className="text-5xl text-netflix-red" />}
              features={[
                "Multi-label genre prediction",
                "Rating classification (G, PG, R)",
                "Content type detection"
              ]}
            />
            <FeatureCard
              title="Smart Recommendations"
              description="Content-based filtering using Sentence-BERT embeddings (768-dim vectors) and FAISS for fast similarity search"
              icon={<FaRobot className="text-5xl text-netflix-red" />}
              features={[
                "Semantic content matching",
                "Vector similarity search",
                "Fast FAISS indexing"
              ]}
            />
            <FeatureCard
              title="ML Insights"
              description="Visualize model predictions, confidence scores, embedding similarities, and feature importance analysis"
              icon={<FaChartLine className="text-5xl text-netflix-red" />}
              features={[
                "Prediction confidence scores",
                "Feature importance charts",
                "Model performance metrics"
              ]}
            />
          </div>
        </div>
      </div>

      {/* Dataset Info */}
      <div className="py-20 px-12">
        <div className="max-w-5xl mx-auto">
          <div className="bg-gradient-to-r from-netflix-red/10 to-purple-900/10 p-12 rounded-2xl border border-netflix-red/30">
            <div className="flex items-center gap-4 mb-8">
              <FaDatabase className="text-5xl text-netflix-red" />
              <div>
                <h3 className="text-3xl font-bold">Netflix Dataset</h3>
                <p className="text-gray-400">Comprehensive movie & TV show catalog for ML training</p>
              </div>
            </div>
            
            <div className="grid md:grid-cols-4 gap-6 mb-8">
              <div className="text-center">
                <div className="text-4xl font-bold text-netflix-red mb-2">8,000+</div>
                <div className="text-gray-400">Movies & Shows</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-netflix-red mb-2">12+</div>
                <div className="text-gray-400">ML Features</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-netflix-red mb-2">768</div>
                <div className="text-gray-400">Embedding Dims</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-netflix-red mb-2">95%</div>
                <div className="text-gray-400">Accuracy</div>
              </div>
            </div>

            <p className="text-gray-300 mb-6">
              The system leverages the Netflix dataset from Kaggle, featuring comprehensive metadata including 
              titles, descriptions, genres, cast, directors, and release dates to power ML-driven insights.
            </p>

            <Link to="/signup" className="btn-primary inline-flex items-center gap-2">
              <FaBrain /> Explore ML Models <FaArrowRight />
            </Link>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="py-20 px-12 bg-netflix-darkGray">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl font-bold mb-6">Ready to explore AI-powered movie analysis?</h2>
          <p className="text-xl text-gray-400 mb-8">
            Sign up to access the full machine learning dashboard with classification, recommendations, 
            and dataset exploration tools
          </p>
          <Link to="/signup" className="btn-primary text-xl py-4 px-12 inline-flex items-center gap-2">
            Get Started <FaArrowRight />
          </Link>
        </div>
      </div>

      {/* Footer */}
      <footer className="py-12 px-12 border-t border-gray-800">
        <div className="max-w-6xl mx-auto text-center text-gray-400">
          <p>&copy; 2025 AetherFlix AI - Movie Content Classification & Recommendation System</p>
          <p className="mt-2">Built with React, FastAPI, XGBoost, Transformers & FAISS</p>
        </div>
      </footer>
    </div>
  )
}

function FeatureCard({ title, description, icon, features }) {
  return (
    <div className="bg-gradient-to-br from-gray-800 to-gray-900 p-8 rounded-xl border border-gray-700 hover:border-netflix-red transition-all">
      <div className="bg-netflix-red/20 w-16 h-16 rounded-lg flex items-center justify-center mb-6">
        {icon}
      </div>
      <h3 className="text-2xl font-bold mb-4">{title}</h3>
      <p className="text-gray-400 mb-6">{description}</p>
      {features && (
        <ul className="text-sm text-gray-500 space-y-2">
          {features.map((feature, idx) => (
            <li key={idx}>• {feature}</li>
          ))}
        </ul>
      )}
    </div>
  )
}
