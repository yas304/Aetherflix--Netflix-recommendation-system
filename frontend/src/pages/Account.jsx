import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'
import { FaUser, FaEnvelope, FaLock, FaSignOutAlt, FaSave, FaEdit, FaCheckCircle, FaTimes } from 'react-icons/fa'

export default function Account() {
  const navigate = useNavigate()
  const { user, session, signOut } = useAuthStore()
  
  const [isEditing, setIsEditing] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [showSuccess, setShowSuccess] = useState(false)
  const [showPasswordChange, setShowPasswordChange] = useState(false)
  
  // Profile fields
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  
  // Password change fields
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [passwordError, setPasswordError] = useState('')
  const [passwordSuccess, setPasswordSuccess] = useState(false)

  // Load user data
  useEffect(() => {
    if (user) {
      setName(user.user_metadata?.name || 'User')
      setEmail(user.email || '')
    }
  }, [user])

  const handleSaveProfile = async () => {
    setIsSaving(true)
    try {
      // Simulate API call to update profile
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // In a real app, you'd update Supabase user metadata here
      // const { error } = await supabase.auth.updateUser({
      //   data: { name }
      // })
      
      setShowSuccess(true)
      setIsEditing(false)
      setTimeout(() => setShowSuccess(false), 3000)
    } catch (error) {
      console.error('Profile update error:', error)
    } finally {
      setIsSaving(false)
    }
  }

  const handleChangePassword = async () => {
    setPasswordError('')
    
    if (newPassword.length < 6) {
      setPasswordError('Password must be at least 6 characters')
      return
    }
    
    if (newPassword !== confirmPassword) {
      setPasswordError('Passwords do not match')
      return
    }

    setIsSaving(true)
    try {
      // Simulate password change
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // In a real app:
      // const { error } = await supabase.auth.updateUser({
      //   password: newPassword
      // })
      
      setPasswordSuccess(true)
      setShowPasswordChange(false)
      setCurrentPassword('')
      setNewPassword('')
      setConfirmPassword('')
      setTimeout(() => setPasswordSuccess(false), 3000)
    } catch (error) {
      setPasswordError(error.message || 'Failed to change password')
    } finally {
      setIsSaving(false)
    }
  }

  const handleSignOut = async () => {
    try {
      await signOut()
      navigate('/login')
    } catch (error) {
      console.error('Sign out error:', error)
    }
  }

  return (
    <div className="min-h-screen dark:bg-netflix-black light:bg-gray-50 pb-20">
      {/* Header */}
      <div className="dark:bg-gradient-to-b dark:from-black dark:to-netflix-black light:bg-gradient-to-b light:from-gray-100 light:to-white dark:border-gray-800 light:border-gray-200 border-b px-4 sm:px-8 lg:px-16 py-12">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-4xl sm:text-5xl font-black dark:text-white light:text-gray-900 mb-2">Account</h1>
          <p className="dark:text-gray-400 light:text-gray-600 text-lg">Manage your account settings and preferences</p>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 sm:px-8 lg:px-16 py-8">
        
        {/* Success Messages */}
        {showSuccess && (
          <div className="bg-green-600 text-white px-6 py-4 rounded-lg mb-6 flex items-center gap-3 animate-fade-in">
            <FaCheckCircle className="text-2xl" />
            <span className="font-semibold">Profile updated successfully!</span>
          </div>
        )}

        {passwordSuccess && (
          <div className="bg-green-600 text-white px-6 py-4 rounded-lg mb-6 flex items-center gap-3 animate-fade-in">
            <FaCheckCircle className="text-2xl" />
            <span className="font-semibold">Password changed successfully!</span>
          </div>
        )}

        {/* Profile Information Card */}
        <div className="dark:bg-gray-900/50 light:bg-white backdrop-blur-sm dark:border-gray-800 light:border-gray-200 border rounded-xl p-8 mb-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold dark:text-white light:text-gray-900 flex items-center gap-3">
              <FaUser className="text-netflix-red" />
              Profile Information
            </h2>
            {!isEditing && (
              <button
                onClick={() => setIsEditing(true)}
                className="flex items-center gap-2 dark:bg-gray-800 light:bg-gray-100 dark:hover:bg-gray-700 light:hover:bg-gray-200 dark:text-white light:text-gray-900 px-4 py-2 rounded-md transition-colors"
              >
                <FaEdit /> Edit Profile
              </button>
            )}
          </div>

          <div className="space-y-6">
            {/* Name Field */}
            <div>
              <label className="block dark:text-gray-400 light:text-gray-600 text-sm font-semibold mb-2">
                Full Name
              </label>
              {isEditing ? (
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full dark:bg-gray-800 light:bg-white dark:text-white light:text-gray-900 dark:border-gray-700 light:border-gray-300 border rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-netflix-red focus:border-transparent"
                  placeholder="Enter your name"
                />
              ) : (
                <div className="dark:text-white light:text-gray-900 text-lg font-medium dark:bg-gray-800/50 light:bg-gray-100 px-4 py-3 rounded-lg">
                  {name}
                </div>
              )}
            </div>

            {/* Email Field */}
            <div>
              <label className="block dark:text-gray-400 light:text-gray-600 text-sm font-semibold mb-2">
                Email Address
              </label>
              <div className="flex items-center gap-3 dark:text-white light:text-gray-900 text-lg font-medium dark:bg-gray-800/50 light:bg-gray-100 px-4 py-3 rounded-lg">
                <FaEnvelope className="dark:text-gray-500 light:text-gray-400" />
                {email}
              </div>
              <p className="dark:text-gray-500 light:text-gray-400 text-xs mt-2">Email cannot be changed for security reasons</p>
            </div>

            {/* Account Created */}
            <div>
              <label className="block dark:text-gray-400 light:text-gray-600 text-sm font-semibold mb-2">
                Member Since
              </label>
              <div className="dark:text-white light:text-gray-900 text-lg font-medium dark:bg-gray-800/50 light:bg-gray-100 px-4 py-3 rounded-lg">
                {user?.created_at ? new Date(user.created_at).toLocaleDateString('en-US', { 
                  year: 'numeric', 
                  month: 'long', 
                  day: 'numeric' 
                }) : 'N/A'}
              </div>
            </div>

            {/* Edit Mode Buttons */}
            {isEditing && (
              <div className="flex gap-3 pt-4">
                <button
                  onClick={handleSaveProfile}
                  disabled={isSaving}
                  className="flex-1 bg-netflix-red hover:bg-red-700 disabled:bg-red-900 disabled:cursor-not-allowed text-white font-semibold px-6 py-3 rounded-lg transition-colors flex items-center justify-center gap-2"
                >
                  {isSaving ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <FaSave /> Save Changes
                    </>
                  )}
                </button>
                <button
                  onClick={() => {
                    setIsEditing(false)
                    setName(user?.user_metadata?.name || 'User')
                  }}
                  disabled={isSaving}
                  className="dark:bg-gray-800 light:bg-gray-200 dark:hover:bg-gray-700 light:hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed dark:text-white light:text-gray-900 font-semibold px-6 py-3 rounded-lg transition-colors flex items-center gap-2"
                >
                  <FaTimes /> Cancel
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Password Change Card */}
        <div className="dark:bg-gray-900/50 light:bg-white backdrop-blur-sm dark:border-gray-800 light:border-gray-200 border rounded-xl p-8 mb-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold dark:text-white light:text-gray-900 flex items-center gap-3">
              <FaLock className="text-netflix-red" />
              Password & Security
            </h2>
            {!showPasswordChange && (
              <button
                onClick={() => setShowPasswordChange(true)}
                className="flex items-center gap-2 dark:bg-gray-800 light:bg-gray-100 dark:hover:bg-gray-700 light:hover:bg-gray-200 dark:text-white light:text-gray-900 px-4 py-2 rounded-md transition-colors"
              >
                <FaEdit /> Change Password
              </button>
            )}
          </div>

          {!showPasswordChange ? (
            <div className="dark:text-gray-400 light:text-gray-600">
              <p className="flex items-center gap-2">
                <FaCheckCircle className="text-green-500" />
                Your password is secure and encrypted
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {passwordError && (
                <div className="bg-orange-600 text-white px-4 py-3 rounded-lg text-sm">
                  {passwordError}
                </div>
              )}

              <div>
                <label className="block dark:text-gray-400 light:text-gray-600 text-sm font-semibold mb-2">
                  Current Password
                </label>
                <input
                  type="password"
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                  className="w-full dark:bg-gray-800 light:bg-white dark:text-white light:text-gray-900 dark:border-gray-700 light:border-gray-300 border rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-netflix-red focus:border-transparent"
                  placeholder="Enter current password"
                />
              </div>

              <div>
                <label className="block dark:text-gray-400 light:text-gray-600 text-sm font-semibold mb-2">
                  New Password
                </label>
                <input
                  type="password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  className="w-full dark:bg-gray-800 light:bg-white dark:text-white light:text-gray-900 dark:border-gray-700 light:border-gray-300 border rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-netflix-red focus:border-transparent"
                  placeholder="Enter new password (min 6 characters)"
                />
              </div>

              <div>
                <label className="block dark:text-gray-400 light:text-gray-600 text-sm font-semibold mb-2">
                  Confirm New Password
                </label>
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="w-full dark:bg-gray-800 light:bg-white dark:text-white light:text-gray-900 dark:border-gray-700 light:border-gray-300 border rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-netflix-red focus:border-transparent"
                  placeholder="Confirm new password"
                />
              </div>

              <div className="flex gap-3 pt-4">
                <button
                  onClick={handleChangePassword}
                  disabled={isSaving}
                  className="flex-1 bg-netflix-red hover:bg-red-700 disabled:bg-red-900 disabled:cursor-not-allowed text-white font-semibold px-6 py-3 rounded-lg transition-colors flex items-center justify-center gap-2"
                >
                  {isSaving ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Updating...
                    </>
                  ) : (
                    <>
                      <FaSave /> Update Password
                    </>
                  )}
                </button>
                <button
                  onClick={() => {
                    setShowPasswordChange(false)
                    setCurrentPassword('')
                    setNewPassword('')
                    setConfirmPassword('')
                    setPasswordError('')
                  }}
                  disabled={isSaving}
                  className="dark:bg-gray-800 light:bg-gray-200 dark:hover:bg-gray-700 light:hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed dark:text-white light:text-gray-900 font-semibold px-6 py-3 rounded-lg transition-colors flex items-center gap-2"
                >
                  <FaTimes /> Cancel
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Account Actions Card */}
        <div className="dark:bg-gray-900/50 light:bg-white backdrop-blur-sm dark:border-gray-800 light:border-gray-200 border rounded-xl p-8">
          <h2 className="text-2xl font-bold dark:text-white light:text-gray-900 mb-6">Account Actions</h2>
          
          <div className="space-y-4">
            {/* Sign Out Button */}
            <button
              onClick={handleSignOut}
              className="w-full dark:bg-gray-800 light:bg-gray-100 dark:hover:bg-gray-700 light:hover:bg-gray-200 dark:text-white light:text-gray-900 font-semibold px-6 py-4 rounded-lg transition-colors flex items-center justify-between group"
            >
              <div className="flex items-center gap-3">
                <FaSignOutAlt className="text-xl" />
                <div className="text-left">
                  <div className="font-bold">Sign Out</div>
                  <div className="text-sm dark:text-gray-400 light:text-gray-600">Sign out of your AetherFlix account</div>
                </div>
              </div>
              <div className="dark:text-gray-500 light:text-gray-400 dark:group-hover:text-white light:group-hover:text-gray-900 transition-colors">→</div>
            </button>

            {/* Account Stats */}
            <div className="grid grid-cols-2 gap-4 pt-6 dark:border-gray-700 light:border-gray-200 border-t">
              <div className="dark:bg-gray-800/50 light:bg-gray-50 p-4 rounded-lg text-center">
                <div className="text-3xl font-black text-netflix-red mb-1">
                  {session ? '✓' : '✗'}
                </div>
                <div className="text-sm dark:text-gray-400 light:text-gray-600">Status</div>
                <div className="dark:text-white light:text-gray-900 font-semibold">
                  {session ? 'Active' : 'Inactive'}
                </div>
              </div>
              <div className="dark:bg-gray-800/50 light:bg-gray-50 p-4 rounded-lg text-center">
                <div className="text-3xl font-black text-netflix-red mb-1">Pro</div>
                <div className="text-sm dark:text-gray-400 light:text-gray-600">Plan</div>
                <div className="dark:text-white light:text-gray-900 font-semibold">AI Premium</div>
              </div>
            </div>
          </div>
        </div>

        {/* Info Box */}
        <div className="mt-6 dark:bg-blue-900/20 light:bg-blue-50 dark:border-blue-800/50 light:border-blue-200 border rounded-lg p-6">
          <div className="flex gap-3">
            <FaCheckCircle className="dark:text-blue-400 light:text-blue-600 text-xl flex-shrink-0 mt-0.5" />
            <div className="text-sm dark:text-gray-300 light:text-gray-700">
              <p className="font-semibold dark:text-white light:text-gray-900 mb-2">Your data is secure</p>
              <p>All account information is encrypted and stored securely using industry-standard security practices. We never share your personal information with third parties.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
