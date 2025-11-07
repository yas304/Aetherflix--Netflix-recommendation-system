import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'

export default function Login() {
  const navigate = useNavigate()
  const { signIn } = useAuthStore()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [rememberMe, setRememberMe] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      await signIn(email, password)
      navigate('/browse')
    } catch (err) {
      setError(err.message || 'Incorrect email or password.')
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

      {/* Login Form Container */}
      <div className="relative z-10 flex items-center justify-center px-4 pt-8 pb-32">
        <div className="w-full max-w-md">
          {/* Main Login Card */}
          <div className="bg-black/75 backdrop-blur-sm rounded-md px-12 py-12 sm:px-16 sm:py-16">
            <h2 className="text-white text-3xl font-bold mb-7">Sign In</h2>

            {error && (
              <div className="bg-orange-600 text-white px-4 py-3 rounded mb-4 text-sm">
                {error}
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="relative">
                <input
                  type="email"
                  placeholder="Email or phone number"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full bg-gray-700 text-white border border-gray-600 rounded px-5 py-4 focus:outline-none focus:ring-2 focus:ring-white focus:bg-gray-600 placeholder:text-gray-400"
                  required
                />
              </div>

              <div className="relative">
                <input
                  type="password"
                  placeholder="Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full bg-gray-700 text-white border border-gray-600 rounded px-5 py-4 focus:outline-none focus:ring-2 focus:ring-white focus:bg-gray-600 placeholder:text-gray-400"
                  required
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-netflix-red hover:bg-red-700 text-white font-semibold py-3.5 rounded transition-colors disabled:bg-red-900 disabled:cursor-not-allowed mt-6"
              >
                {loading ? 'Signing In...' : 'Sign In'}
              </button>

              <div className="flex items-center justify-between text-sm text-gray-400 mt-3">
                <label className="flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={rememberMe}
                    onChange={(e) => setRememberMe(e.target.checked)}
                    className="mr-2 w-4 h-4"
                  />
                  <span>Remember me</span>
                </label>
                <a href="#" className="hover:underline">Need help?</a>
              </div>
            </form>

            {/* Sign Up Link */}
            <div className="mt-8 text-gray-400">
              <span>New to AetherFlix? </span>
              <Link to="/signup" className="text-white hover:underline font-medium">
                Sign up now
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
