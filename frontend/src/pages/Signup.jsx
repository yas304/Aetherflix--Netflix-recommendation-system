import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'
import { FaCheckCircle } from 'react-icons/fa'

export default function Signup() {
  const navigate = useNavigate()
  const { signUp } = useAuthStore()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [name, setName] = useState('')
  const [error, setError] = useState('')
  const [success, setSuccess] = useState(false)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    if (password.length < 6) {
      setError('Password should be at least 6 characters.')
      setLoading(false)
      return
    }

    try {
      await signUp(email, password, { name })
      setSuccess(true)
      setTimeout(() => {
        navigate('/login')
      }, 2000)
    } catch (err) {
      setError(err.message || 'Failed to create account. Email may already exist.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-netflix-black relative overflow-hidden">
      {/* Netflix-style Background */}
      <div className="absolute inset-0 z-0">
        <div className="absolute inset-0 bg-gradient-to-b from-black/60 via-black/50 to-netflix-black" />
        <img
          src="https://assets.nflxext.com/ffe/siteui/vlv3/fc164b4b-f085-44ee-bb7f-ec7df8539eff/d23a1608-7d90-4da1-93d6-bae2fe60a69b/IN-en-20230814-popsignuptwoweeks-perspective_alpha_website_large.jpg"
          alt="Background"
          className="w-full h-full object-cover"
        />
      </div>

      {/* Signup Form Container */}
      <div className="relative z-10 flex items-center justify-center px-4 pt-8 pb-32">
        <div className="w-full max-w-md">
          {/* Main Signup Card */}
          <div className="bg-black/75 backdrop-blur-sm rounded-md px-12 py-12 sm:px-16 sm:py-16">
            <h2 className="text-white text-3xl font-bold mb-3">Create Account</h2>
            <p className="text-gray-400 text-sm mb-7">
              Join AetherFlix AI to get personalized movie recommendations powered by ML.
            </p>

            {error && (
              <div className="bg-orange-600 text-white px-4 py-3 rounded mb-4 text-sm">
                {error}
              </div>
            )}

            {success && (
              <div className="bg-green-600 text-white px-4 py-3 rounded mb-4 text-sm flex items-center gap-2">
                <FaCheckCircle />
                <span>Account created! Redirecting to login...</span>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="relative">
                <input
                  type="text"
                  placeholder="Full Name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full bg-gray-700 text-white border border-gray-600 rounded px-5 py-4 focus:outline-none focus:ring-2 focus:ring-white focus:bg-gray-600 placeholder:text-gray-400"
                  required
                />
              </div>

              <div className="relative">
                <input
                  type="email"
                  placeholder="Email address"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full bg-gray-700 text-white border border-gray-600 rounded px-5 py-4 focus:outline-none focus:ring-2 focus:ring-white focus:bg-gray-600 placeholder:text-gray-400"
                  required
                />
              </div>

              <div className="relative">
                <input
                  type="password"
                  placeholder="Password (min. 6 characters)"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full bg-gray-700 text-white border border-gray-600 rounded px-5 py-4 focus:outline-none focus:ring-2 focus:ring-white focus:bg-gray-600 placeholder:text-gray-400"
                  minLength={6}
                  required
                />
              </div>

              <button
                type="submit"
                disabled={loading || success}
                className="w-full bg-netflix-red hover:bg-red-700 text-white font-semibold py-3.5 rounded transition-colors disabled:bg-red-900 disabled:cursor-not-allowed mt-6"
              >
                {loading ? 'Creating Account...' : success ? 'Account Created!' : 'Sign Up'}
              </button>
            </form>

            {/* Benefits */}
            <div className="mt-8 pt-6 border-t border-gray-700">
              <p className="text-gray-400 text-xs mb-3">By signing up, you get:</p>
              <ul className="space-y-2 text-sm text-gray-300">
                <li className="flex items-start gap-2">
                  <FaCheckCircle className="text-green-500 mt-0.5 flex-shrink-0" />
                  <span>AI-powered movie & TV show recommendations</span>
                </li>
                <li className="flex items-start gap-2">
                  <FaCheckCircle className="text-green-500 mt-0.5 flex-shrink-0" />
                  <span>Personalized content discovery by genre</span>
                </li>
                <li className="flex items-start gap-2">
                  <FaCheckCircle className="text-green-500 mt-0.5 flex-shrink-0" />
                  <span>Smart content filtering and ratings</span>
                </li>
              </ul>
            </div>

            {/* Sign In Link */}
            <div className="mt-6 text-gray-400">
              <span>Already have an account? </span>
              <Link to="/login" className="text-white hover:underline font-medium">
                Sign in
              </Link>
              .
            </div>

            <div className="mt-4 text-xs text-gray-500">
              This page is protected by Google reCAPTCHA to ensure you're not a bot.{' '}
              <a href="#" className="text-blue-500 hover:underline">Learn more</a>.
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="relative z-10 bg-black/80 border-t border-gray-800 px-8 sm:px-12 lg:px-16 py-8">
        <div className="max-w-4xl mx-auto text-gray-500 text-sm">
          <p className="mb-4">Questions? Contact us.</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
            <a href="#" className="hover:underline">FAQ</a>
            <a href="#" className="hover:underline">Help Center</a>
            <a href="#" className="hover:underline">Terms of Use</a>
            <a href="#" className="hover:underline">Privacy</a>
            <a href="#" className="hover:underline">Cookie Preferences</a>
            <a href="#" className="hover:underline">Corporate Information</a>
          </div>
          <p className="text-xs">&copy; 2025 AetherFlix AI. All rights reserved.</p>
        </div>
      </div>
    </div>
  )
}
