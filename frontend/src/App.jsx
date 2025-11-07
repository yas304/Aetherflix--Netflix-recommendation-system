import { useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuthStore } from './store/authStore'

// Pages
import Landing from './pages/Landing'
import Login from './pages/Login'
import Signup from './pages/Signup'
import Browse from './pages/Browse'
import Search from './pages/Search'
import MyList from './pages/MyList'
import Account from './pages/Account'

// Layout
import Layout from './components/Layout/Layout'

function App() {
  const { session, loading, initialize } = useAuthStore()

  // Apply dark theme to document (fixed theme)
  useEffect(() => {
    document.documentElement.classList.remove('light')
    document.documentElement.classList.add('dark')
  }, [])

  // Initialize auth on mount
  useEffect(() => {
    // Set a timeout to prevent infinite loading
    const timeout = setTimeout(() => {
      if (loading) {
        console.log('Auth initialization timeout - proceeding without auth')
      }
    }, 3000)
    
    initialize()
    
    return () => clearTimeout(timeout)
  }, [initialize])

  // Show loading for max 3 seconds
  if (loading) {
    return (
      <div className="min-h-screen bg-netflix-black flex items-center justify-center">
        <div className="text-center">
          <div className="text-6xl font-black text-netflix-red animate-pulse mb-4">
            AETHERFLIX
          </div>
          <div className="text-gray-400 text-lg">AI-Powered Content Platform</div>
        </div>
      </div>
    )
  }

  return (
    <Routes>
      {/* Public routes - No auth required for ML demo */}
      <Route path="/" element={<Landing />} />
      <Route path="/login" element={session ? <Navigate to="/browse" /> : <Login />} />
      <Route path="/signup" element={session ? <Navigate to="/browse" /> : <Signup />} />
      
      {/* Browse page - No auth required for ML demo */}
      <Route path="/browse" element={
        <Layout>
          <Browse />
        </Layout>
      } />

      {/* Protected routes - require auth */}
      <Route
        path="/*"
        element={
          session ? (
            <Layout>
              <Routes>
                <Route path="/search" element={<Search />} />
                <Route path="/my-list" element={<MyList />} />
                <Route path="/account" element={<Account />} />
              </Routes>
            </Layout>
          ) : (
            <Navigate to="/browse" />
          )
        }
      />
    </Routes>
  )
}

export default App
