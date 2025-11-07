import { useState } from 'react'
import { FaBrain, FaStar, FaFilm, FaTv } from 'react-icons/fa'
import api from '../lib/api'

export default function Browse() {
  const [favoriteTitle, setFavoriteTitle] = useState('')
  const [selectedGenre, setSelectedGenre] = useState('')
  const [contentType, setContentType] = useState('Movie')
  const [recommendations, setRecommendations] = useState([])
  const [genreResults, setGenreResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('recommendations')

  const genres = [
    'Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Thriller',
    'Sci-Fi', 'Documentary', 'Animation', 'Crime', 'Fantasy', 'Adventure'
  ]

  const getRecommendations = async () => {
    if (!favoriteTitle.trim()) return
    setLoading(true)
    try {
      const res = await api.post('/api/recommend', { 
        title: favoriteTitle,
        top_n: 10 
      })
      setRecommendations(res.data.recommendations || [])
      setActiveTab('recommendations')
    } catch (err) {
      console.error('Recommendation error:', err)
      alert(err.response?.data?.detail || 'Title not found. Try another title!')
    } finally {
      setLoading(false)
    }
  }

  const browseByGenre = async (genre) => {
    setSelectedGenre(genre)
    setLoading(true)
    try {
      const res = await api.get(`/api/browse?type=${contentType}&limit=100`)
      const results = res.data.results || []
      
      const filtered = results
        .filter(item => item.listed_in?.includes(genre))
        .sort((a, b) => {
          const ratingOrder = ['TV-MA', 'R', 'TV-14', 'PG-13', 'PG', 'TV-G', 'G']
          const aIndex = ratingOrder.indexOf(a.rating) !== -1 ? ratingOrder.indexOf(a.rating) : 999
          const bIndex = ratingOrder.indexOf(b.rating) !== -1 ? ratingOrder.indexOf(b.rating) : 999
          return aIndex - bIndex
        })
        .slice(0, 20)
      
      setGenreResults(filtered)
      setActiveTab('genre')
    } catch (err) {
      console.error('Genre browse error:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen dark:bg-netflix-black light:bg-gray-50 pb-20">
      <div className="relative h-[70vh] dark:bg-gradient-to-b dark:from-black dark:via-black/80 dark:to-netflix-black light:bg-gradient-to-b light:from-gray-100 light:via-gray-200 light:to-gray-50">
        <div 
          className="absolute inset-0 bg-cover bg-center opacity-30" 
          style={{ backgroundImage: 'url(https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=1920&q=80)' }} 
        />
        <div className="absolute inset-0 dark:bg-gradient-to-t dark:from-netflix-black dark:via-netflix-black/60 light:bg-gradient-to-t light:from-gray-50 light:via-gray-50/60 to-transparent" />
        
        <div className="relative h-full flex flex-col justify-center px-4 sm:px-8 lg:px-16 max-w-7xl mx-auto">
          <div className="flex items-center gap-4 mb-6 animate-fade-in">
            <FaBrain className="text-5xl sm:text-6xl text-netflix-red drop-shadow-2xl" />
            <div>
              <h1 className="text-5xl sm:text-7xl font-black tracking-tight dark:text-white light:text-gray-900">
                Smart Recommendations
              </h1>
              <p className="text-netflix-red text-lg sm:text-xl font-bold mt-1">AETHERFLIX AI</p>
            </div>
          </div>
          
          <p className="text-xl sm:text-2xl dark:text-gray-200 light:text-gray-700 mb-8 max-w-3xl leading-relaxed font-medium">
            Get personalized recommendations based on your favorite shows or browse by genre with ratings.
          </p>

          <div className="flex gap-4 mb-6">
            <button
              onClick={() => setContentType('Movie')}
              className={`px-6 py-3 rounded-lg font-bold text-lg flex items-center gap-2 transition-all ${
                contentType === 'Movie' 
                  ? 'bg-netflix-red text-white' 
                  : 'dark:bg-white/20 light:bg-gray-200 dark:text-gray-300 light:text-gray-700 dark:hover:bg-white/30 light:hover:bg-gray-300'
              }`}
            >
              <FaFilm /> Movies
            </button>
            <button
              onClick={() => setContentType('TV Show')}
              className={`px-6 py-3 rounded-lg font-bold text-lg flex items-center gap-2 transition-all ${
                contentType === 'TV Show' 
                  ? 'bg-netflix-red text-white' 
                  : 'dark:bg-white/20 light:bg-gray-200 dark:text-gray-300 light:text-gray-700 dark:hover:bg-white/30 light:hover:bg-gray-300'
              }`}
            >
              <FaTv /> TV Shows
            </button>
          </div>

          <div className="dark:bg-black/70 light:bg-white/90 backdrop-blur-md p-6 sm:p-8 rounded-lg dark:border-gray-700/50 light:border-gray-200 border max-w-4xl shadow-2xl">
            <h3 className="text-2xl font-bold mb-4 text-netflix-red">Get Recommendations</h3>
            <div className="flex flex-col sm:flex-row gap-4">
              <input
                value={favoriteTitle}
                onChange={(e) => setFavoriteTitle(e.target.value)}
                placeholder={`Enter your favorite ${contentType.toLowerCase()} (e.g., "${contentType === 'Movie' ? 'The Dark Knight' : 'Breaking Bad'}")`}
                className="flex-1 dark:bg-gray-900/90 light:bg-white dark:text-white light:text-gray-900 px-6 py-4 rounded-md dark:border-gray-600 light:border-gray-300 border dark:focus:border-netflix-red light:focus:border-netflix-red focus:outline-none focus:ring-2 focus:ring-netflix-red/50 transition-all dark:placeholder:text-gray-500 light:placeholder:text-gray-400 text-lg"
                onKeyPress={(e) => e.key === 'Enter' && getRecommendations()}
              />
              <button 
                onClick={getRecommendations} 
                disabled={loading || !favoriteTitle.trim()}
                className="bg-netflix-red hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed px-8 py-4 rounded-md font-black text-lg text-white flex items-center justify-center gap-3 transition-all transform hover:scale-105 active:scale-95 shadow-lg whitespace-nowrap"
              >
                {loading ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Finding...
                  </>
                ) : (
                  <>
                    <FaBrain className="text-xl" /> Get 10 Similar
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="px-4 sm:px-8 lg:px-16 py-12 max-w-7xl mx-auto">
        <h2 className="text-3xl sm:text-4xl font-black mb-6 flex items-center gap-3 dark:text-white light:text-gray-900">
          <FaStar className="text-netflix-red" />
          Browse by Genre (Sorted by Rating)
        </h2>
        
        <div className="flex flex-wrap gap-3 mb-12">
          {genres.map((genre) => (
            <button
              key={genre}
              onClick={() => browseByGenre(genre)}
              disabled={loading}
              className={`px-6 py-3 rounded-full font-bold transition-all ${
                selectedGenre === genre 
                  ? 'bg-netflix-red text-white scale-110' 
                  : 'dark:bg-gray-800 light:bg-gray-200 dark:text-gray-300 light:text-gray-700 dark:hover:bg-gray-700 light:hover:bg-gray-300 hover:scale-105'
              }`}
            >
              {genre}
            </button>
          ))}
        </div>

        {loading && (
          <div className="flex justify-center items-center py-20">
            <div className="text-center">
              <div className="w-16 h-16 border-4 border-netflix-red/30 border-t-netflix-red rounded-full animate-spin mx-auto mb-4" />
              <p className="dark:text-gray-400 light:text-gray-600 text-lg">Loading amazing content...</p>
            </div>
          </div>
        )}

        {!loading && activeTab === 'recommendations' && recommendations.length > 0 && (
          <div className="space-y-6">
            <h3 className="text-2xl font-bold dark:text-white light:text-gray-900 mb-4">
              Because you liked <span className="text-netflix-red">"{favoriteTitle}"</span>
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {recommendations.map((item, index) => (
                <div 
                  key={index}
                  className="dark:bg-gradient-to-br dark:from-gray-900 dark:to-black light:bg-white p-6 rounded-xl dark:border-gray-700 light:border-gray-200 border dark:hover:border-netflix-red light:hover:border-netflix-red transition-all transform hover:scale-[1.02] cursor-pointer"
                >
                  <div className="flex items-start gap-4">
                    <div className="bg-netflix-red text-white font-black text-xl w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0">
                      {index + 1}
                    </div>
                    <div className="flex-1">
                      <h4 className="text-xl font-bold dark:text-white light:text-gray-900 mb-2">{item.title}</h4>
                      <div className="flex items-center gap-3 mb-2 text-sm">
                        <span className="text-green-400 font-semibold">
                          {item.type === 'Movie' ? '🎬 Movie' : '📺 TV Show'}
                        </span>
                        {item.release_year && (
                          <span className="dark:text-gray-400 light:text-gray-600">{item.release_year}</span>
                        )}
                        {item.rating && (
                          <span className="dark:border-gray-500 light:border-gray-300 border px-2 py-0.5 text-xs dark:text-gray-300 light:text-gray-700">
                            {item.rating}
                          </span>
                        )}
                      </div>
                      {item.listed_in && (
                        <p className="dark:text-gray-400 light:text-gray-600 text-sm mb-2">
                          {item.listed_in.split(',').slice(0, 3).join(' • ')}
                        </p>
                      )}
                      {item.description && (
                        <p className="dark:text-gray-500 light:text-gray-500 text-sm line-clamp-2">
                          {item.description}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {!loading && activeTab === 'genre' && genreResults.length > 0 && (
          <div className="space-y-6">
            <h3 className="text-2xl font-bold dark:text-white light:text-gray-900 mb-4">
              <span className="text-netflix-red">{selectedGenre}</span> {contentType}s (Highest Rated First)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {genreResults.map((item, index) => (
                <div 
                  key={index}
                  className="dark:bg-gradient-to-br dark:from-gray-900 dark:to-black light:bg-white p-5 rounded-xl dark:border-gray-700 light:border-gray-200 border dark:hover:border-netflix-red light:hover:border-netflix-red transition-all transform hover:scale-[1.02] cursor-pointer"
                >
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="text-lg font-bold dark:text-white light:text-gray-900 line-clamp-1">{item.title}</h4>
                    {item.rating && (
                      <span className="bg-netflix-red px-3 py-1 rounded-full text-xs font-bold text-white">
                        {item.rating}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2 mb-2 text-sm">
                    {item.release_year && (
                      <span className="dark:text-gray-400 light:text-gray-600">{item.release_year}</span>
                    )}
                    <span className="text-green-400 text-xs">
                      {item.type === 'Movie' ? '🎬' : '📺'}
                    </span>
                  </div>
                  {item.description && (
                    <p className="dark:text-gray-500 light:text-gray-500 text-sm line-clamp-3">
                      {item.description}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {!loading && activeTab === 'recommendations' && recommendations.length === 0 && favoriteTitle && (
          <div className="text-center py-20">
            <FaBrain className="text-6xl dark:text-gray-700 light:text-gray-300 mx-auto mb-4" />
            <p className="dark:text-gray-400 light:text-gray-600 text-lg">No recommendations found. Try a different title!</p>
          </div>
        )}

        {!loading && activeTab === 'genre' && genreResults.length === 0 && selectedGenre && (
          <div className="text-center py-20">
            <FaStar className="text-6xl dark:text-gray-700 light:text-gray-300 mx-auto mb-4" />
            <p className="dark:text-gray-400 light:text-gray-600 text-lg">No {contentType.toLowerCase()}s found for {selectedGenre}. Try another genre!</p>
          </div>
        )}
      </div>
    </div>
  )
}
