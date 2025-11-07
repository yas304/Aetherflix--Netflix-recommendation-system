import { useState } from 'react'
import { FaSearch } from 'react-icons/fa'
import { useQuery } from '@tanstack/react-query'
import api from '../lib/api'
import ContentCard from '../components/Browse/ContentCard'

export default function Search() {
  const [query, setQuery] = useState('')
  const [searchTerm, setSearchTerm] = useState('')

  const { data: results, isLoading } = useQuery({
    queryKey: ['search', searchTerm],
    queryFn: async () => {
      if (!searchTerm) return null
      const response = await api.post('/api/recommend', {
        query: searchTerm,
        method: 'semantic',
        limit: 20,
      })
      return response.data
    },
    enabled: !!searchTerm,
  })

  const handleSearch = (e) => {
    e.preventDefault()
    setSearchTerm(query)
  }

  return (
    <div className="min-h-screen bg-netflix-black pt-24 px-12">
      {/* Search Bar */}
      <div className="max-w-4xl mx-auto mb-12">
        <form onSubmit={handleSearch} className="relative">
          <input
            type="text"
            placeholder="Search for movies, TV shows, genres..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full bg-netflix-darkGray text-white text-xl px-6 py-4 rounded-lg border-2 border-gray-700 focus:border-white focus:outline-none transition-colors"
          />
          <button
            type="submit"
            className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white transition-colors"
          >
            <FaSearch size={24} />
          </button>
        </form>
      </div>

      {/* Results */}
      {isLoading && (
        <div className="text-center text-gray-400 text-xl">Searching...</div>
      )}

      {results?.recommendations && (
        <div>
          <h2 className="text-2xl font-bold mb-6">
            Search Results for "{searchTerm}"
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
            {results.recommendations.map((item) => (
              <ContentCard key={item.id} content={item} />
            ))}
          </div>

          {results.recommendations.length === 0 && (
            <div className="text-center text-gray-400 text-xl">
              No results found. Try a different search term.
            </div>
          )}
        </div>
      )}

      {!searchTerm && (
        <div className="text-center text-gray-400 text-xl">
          Enter a search term to find content
        </div>
      )}
    </div>
  )
}
