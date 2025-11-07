import { useState } from 'react'
import { FaPlay, FaPlus, FaChevronDown, FaFilm } from 'react-icons/fa'

export default function ContentCard({ content }) {
  const [isHovered, setIsHovered] = useState(false)
  
  // Parse genres from listed_in field
  const genres = content.listed_in ? content.listed_in.split(',').map(g => g.trim()).slice(0, 3) : []
  
  // Generate a placeholder image based on type
  const getPlaceholderImage = () => {
    const colors = ['#E50914', '#831010', '#B20710', '#F40612']
    const randomColor = colors[Math.floor(Math.random() * colors.length)]
    return `https://via.placeholder.com/250x140/${randomColor.substring(1)}/FFFFFF?text=${encodeURIComponent(content.title?.substring(0, 20) || 'Movie')}`
  }

  return (
    <div
      className="relative min-w-[250px] h-[140px] cursor-pointer transition-transform duration-300 hover:scale-105 hover:z-50"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Thumbnail */}
      <div className="w-full h-full bg-gradient-to-br from-gray-800 to-gray-900 rounded-md flex items-center justify-center overflow-hidden">
        <img
          src={getPlaceholderImage()}
          alt={content.title}
          className="w-full h-full object-cover"
          onError={(e) => {
            e.target.style.display = 'none'
            e.target.parentElement.innerHTML = `<div class="flex items-center justify-center w-full h-full"><FaFilm class="text-netflix-red text-4xl" /></div>`
          }}
        />
      </div>

      {/* Hover Overlay */}
      {isHovered && (
        <div className="absolute inset-0 bg-netflix-darkGray rounded-md shadow-2xl p-4 flex flex-col justify-between border-2 border-white/20 animate-fade-in">
          <div>
            <h3 className="font-bold text-base mb-2 line-clamp-2">{content.title}</h3>
            <div className="flex items-center gap-2 text-xs mb-2">
              <span className="text-green-400 font-bold">
                {content.type === 'Movie' ? '🎬 Movie' : '📺 TV Show'}
              </span>
              {content.release_year && (
                <span className="text-gray-400">{content.release_year}</span>
              )}
            </div>
            {content.rating && (
              <span className="inline-block border border-gray-500 px-2 py-0.5 text-xs mb-2">
                {content.rating}
              </span>
            )}
            <div className="flex flex-wrap gap-1 mb-2">
              {genres.slice(0, 2).map((genre, idx) => (
                <span key={idx} className="text-xs text-gray-300">
                  {genre}
                  {idx < genres.slice(0, 2).length - 1 ? ' •' : ''}
                </span>
              ))}
            </div>
            {content.description && (
              <p className="text-xs text-gray-400 line-clamp-2">
                {content.description}
              </p>
            )}
          </div>

          <div className="flex gap-2 mt-2">
            <button className="w-8 h-8 rounded-full bg-white flex items-center justify-center text-black hover:bg-gray-300 transition-colors">
              <FaPlay size={12} />
            </button>
            <button className="w-8 h-8 rounded-full border-2 border-gray-400 flex items-center justify-center hover:border-white transition-colors">
              <FaPlus size={12} />
            </button>
            <button className="w-8 h-8 rounded-full border-2 border-gray-400 flex items-center justify-center hover:border-white transition-colors ml-auto">
              <FaChevronDown size={12} />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
