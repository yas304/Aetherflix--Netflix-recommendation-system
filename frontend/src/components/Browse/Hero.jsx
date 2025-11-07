import { FaPlay, FaInfoCircle } from 'react-icons/fa'

export default function Hero({ content }) {
  if (!content) return null

  return (
    <div className="relative h-screen">
      {/* Background Image */}
      <div className="absolute inset-0">
        <img
          src={content.poster_url}
          alt={content.title}
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-r from-netflix-black via-netflix-black/80 to-transparent" />
        <div className="absolute inset-0 hero-gradient" />
      </div>

      {/* Content */}
      <div className="relative z-10 h-full flex flex-col justify-center px-12 md:px-20 max-w-3xl">
        <h1 className="text-5xl md:text-7xl font-bold mb-4 drop-shadow-lg slide-up">
          {content.title}
        </h1>
        
        <div className="flex items-center gap-4 mb-6 text-lg slide-up" style={{ animationDelay: '0.2s' }}>
          <span className="text-green-500 font-bold">
            {(content.rating * 10).toFixed(0)}% Match
          </span>
          <span>{content.release_year}</span>
          <span className="border border-gray-400 px-2 py-0.5 text-sm">
            {content.rating >= 8.5 ? 'TV-MA' : 'TV-14'}
          </span>
        </div>

        <p className="text-lg md:text-xl mb-8 line-clamp-3 drop-shadow-lg slide-up" style={{ animationDelay: '0.4s' }}>
          {content.description}
        </p>

        <div className="flex gap-4 slide-up" style={{ animationDelay: '0.6s' }}>
          <button className="btn-primary flex items-center gap-2 text-xl py-3 px-8">
            <FaPlay /> Play
          </button>
          <button className="btn-secondary flex items-center gap-2 text-xl py-3 px-8">
            <FaInfoCircle /> More Info
          </button>
        </div>

        <div className="mt-8 flex gap-2 slide-up" style={{ animationDelay: '0.8s' }}>
          {content.genres?.map((genre) => (
            <span key={genre} className="text-sm text-gray-300">
              {genre}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}
